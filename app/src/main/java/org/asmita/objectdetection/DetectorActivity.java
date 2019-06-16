/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.asmita.objectdetection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

import org.asmita.objectdetection.customview.OverlayView;
import org.asmita.objectdetection.customview.OverlayView.DrawCallback;
import org.asmita.objectdetection.env.BorderedText;
import org.asmita.objectdetection.env.ImageUtils;
import org.asmita.objectdetection.env.Logger;
import org.asmita.objectdetection.tflite.Classifier;
import org.asmita.objectdetection.tflite.TFLiteObjectDetectionAPIModel;
import org.asmita.objectdetection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  private static final int MIN_STALE_SILENT_DURATION = 5000;

  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;
  TextToSpeech tts;
  private boolean canSpeak = false;
  private long lastSpokenTimeStamp = 0;
  private String lastSpokenSentence = "";

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    tts = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
      @Override
      public void onInit(int status) {
        if(status != TextToSpeech.ERROR) {
          tts.setLanguage(Locale.UK);
          canSpeak = true;
        } else {
          Log.e("text to speech", "initialization failed");
        }
      }
    });
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  private String objectPositionClassifier(Classifier.Recognition object) {
    int frameWidth = croppedBitmap.getWidth();
    float objectLeft = object.getLocation().left;
    float objectRight = object.getLocation().right;

    double frameCentre = frameWidth*0.5;
    double rightEdgeDistanceFromCentre = (objectRight - frameCentre);
    double leftEdgeDistanceFromCentre = (objectLeft - frameCentre);

    // left and right edge of object is on different sides of the centre -> object is in front
    if (rightEdgeDistanceFromCentre * leftEdgeDistanceFromCentre  < 0) {
      return "front";
    } else if (rightEdgeDistanceFromCentre < 0){
      if (Math.abs(rightEdgeDistanceFromCentre) <= frameWidth*0.05
              && Math.abs(leftEdgeDistanceFromCentre) <= frameWidth*0.15) {
        return "front";
      }
      return "left";
    } else {
      if (Math.abs(rightEdgeDistanceFromCentre) <= frameWidth*0.15
              && Math.abs(leftEdgeDistanceFromCentre) <= frameWidth*0.05) {
        return "front";
      }
      return "right";
    }
  }

  private HashMap<String, ArrayList<String>> getPartitionedObjects(List<Classifier.Recognition> results) {
    HashMap<String, ArrayList<String>> partitionedObjects = new HashMap<>();
    partitionedObjects.put("left", new ArrayList<>());
    partitionedObjects.put("front", new ArrayList<>());
    partitionedObjects.put("right", new ArrayList<>());

    for (Classifier.Recognition result: results) {
      String position = objectPositionClassifier(result);
      partitionedObjects.get(position).add(result.getTitle());
    }
    return partitionedObjects;
  }

  private String formPartialSentence(ArrayList<String> objects) {
    if (objects == null || objects.size() == 0) {
      return "";
    }
    String partialSentence = String.join(", ", objects);
    // replace last comma with 'and'
    // ex: book, laptop, apple -> book, laptop and apple
    int lastCommaIndex = partialSentence.lastIndexOf(",");

    if (lastCommaIndex == -1) {
      return partialSentence;
    }
    return partialSentence.substring(0, lastCommaIndex)+ " and" + partialSentence.substring(lastCommaIndex);
  }

  private String formSentence(String leftSentence, String frontSentence, String rightSentence) {
    if (leftSentence.isEmpty() && frontSentence.isEmpty() && rightSentence.isEmpty()) {
      return "";
    }

    String baseSentence = "There's ";
    boolean previousSentenceExists = false;

    if (!leftSentence.isEmpty()) {
      baseSentence += leftSentence + " on your left, ";
      previousSentenceExists = true;
    }

    if (!frontSentence.isEmpty()) {
      if (previousSentenceExists) {
        baseSentence += "and ";
      }
      baseSentence += frontSentence + " in front of you, ";
      previousSentenceExists = true;
    }

    if (!leftSentence.isEmpty()) {
      if (previousSentenceExists) {
        baseSentence += "and ";
      }
      baseSentence += leftSentence + " on your right";
    }
    // clean trailing ', ' if any
    return baseSentence.replaceAll(", $", "");
  }

  private long getCurrentTimeStamp() {
    Date date = new Date();
    return date.getTime();
  }

  private void sayDetectedObjectLocations(List<Classifier.Recognition> results) {
    HashMap<String, ArrayList<String>> partitionedObjects = getPartitionedObjects(results);
    String objectsOnLeft = formPartialSentence(partitionedObjects.get("left"));
    String objectsInFront = formPartialSentence(partitionedObjects.get("front"));
    String objectsOnRight = formPartialSentence(partitionedObjects.get("right"));

    String completeSentence = formSentence(objectsOnLeft, objectsInFront, objectsOnRight);

    // don't speak same thing again unless you've been silent for 5secs
    if (lastSpokenTimeStamp + MIN_STALE_SILENT_DURATION > getCurrentTimeStamp()
            && lastSpokenSentence.equals(completeSentence)) {
      return;
    }

    lastSpokenTimeStamp = getCurrentTimeStamp();
    lastSpokenSentence = completeSentence;

    Log.d("gonna speak", completeSentence);
    if(canSpeak) {
      tts.speak(completeSentence, TextToSpeech.QUEUE_FLUSH, null);
    } else {
      Log.e("tts error", "cannot speak");
    }
  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);

            sayDetectedObjectLocations(results);

            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}
