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

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.AudioManager;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import androidx.annotation.NonNull;

import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.text.FirebaseVisionText;
import com.google.firebase.ml.vision.text.FirebaseVisionTextRecognizer;
import com.google.firebase.ml.vision.text.RecognizedLanguage;

import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.asmita.objectdetection.customview.OverlayView;
import org.asmita.objectdetection.customview.OverlayView.DrawCallback;
import org.asmita.objectdetection.env.BorderedText;
import org.asmita.objectdetection.env.ImageUtils;
import org.asmita.objectdetection.env.Logger;
import org.asmita.objectdetection.tflite.Classifier;
import org.asmita.objectdetection.tflite.TFLiteObjectDetectionAPIModel;
import org.asmita.objectdetection.tracking.MultiBoxTracker;

import com.google.firebase.ml.vision.barcode.FirebaseVisionBarcode;
import com.google.firebase.ml.vision.barcode.FirebaseVisionBarcodeDetector;
import com.google.firebase.ml.vision.barcode.FirebaseVisionBarcodeDetectorOptions;

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
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.659f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  private static final int MIN_STALE_SILENT_DURATION = 5000;
  private static final String POSITION_LEFT = "left";
  private static final String POSITION_FRONT = "front";
  private static final String POSITION_RIGHT = "right";
  private final String REGEX_OCR_TRIGGER_SPEECH = "(what('s| is) written in front of me)|(read)";
  private final String REGEX_START_GUIDING = "(start guiding( ?me)?)|(guide( ?me)?)";
  private final String REGEX_STOP_GUIDING = "stop( guiding( ?me)?)?";
  private final String ERROR_COULDNT_READ = "I can't see anything written in front of you!";

  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;
  private boolean runningTextRecognition = false;
  private boolean runningBarRecognition = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;
  TextToSpeech tts;
  private boolean canSpeak = false;
  private long lastSpokenTimeStamp = 0;
  private HashMap<String, ArrayList<String>> objectsToSpeak;
  private String detectedText = "";
  private String extractedBarcodeText = "";
  private TextView recognitionResults;
  private TextView barcodeRecognitionResults;
  private boolean shouldGuide = false;

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
    recognitionResults = findViewById(R.id.recognition_results);
    barcodeRecognitionResults = findViewById(R.id.barcode_recognition_results);
    recognitionResults.setMovementMethod(new ScrollingMovementMethod());
    resetObjectsToSpeak();
  }

  private void resetObjectsToSpeak() {
    objectsToSpeak = new HashMap<>();
    objectsToSpeak.put(POSITION_LEFT, new ArrayList<>());
    objectsToSpeak.put(POSITION_FRONT, new ArrayList<>());
    objectsToSpeak.put(POSITION_RIGHT, new ArrayList<>());
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
      return POSITION_FRONT;
    } else if (rightEdgeDistanceFromCentre < 0){
      if (Math.abs(rightEdgeDistanceFromCentre) <= frameWidth*0.05
              && Math.abs(leftEdgeDistanceFromCentre) <= frameWidth*0.15) {
        return POSITION_FRONT;
      }
      return POSITION_LEFT;
    } else {
      if (Math.abs(rightEdgeDistanceFromCentre) <= frameWidth*0.15
              && Math.abs(leftEdgeDistanceFromCentre) <= frameWidth*0.05) {
        return POSITION_FRONT;
      }
      return POSITION_RIGHT;
    }
  }

  private HashMap<String, ArrayList<String>> getobjectsToSpeak(List<Classifier.Recognition> results) {
    for (Classifier.Recognition result: results) {
      String position = objectPositionClassifier(result);
      ArrayList<String> objectsInPosition = objectsToSpeak.get(position);
      // don't add same object to same position again
      if (!objectsInPosition.contains(result.getTitle())) {
        objectsInPosition.add(result.getTitle());
      }
    }
    return objectsToSpeak;
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
    return partialSentence.substring(0, lastCommaIndex)+ " and" + partialSentence.substring(lastCommaIndex+1);
  }

  private String formSentence(String leftSentence, String frontSentence, String rightSentence) {
    if (leftSentence.isEmpty() && frontSentence.isEmpty() && rightSentence.isEmpty()) {
      return "";
    }

    String baseSentence = "There's ";
    boolean previousSentenceExists = false;

    if (!leftSentence.isEmpty()) {
      baseSentence += leftSentence + " on your left";
      previousSentenceExists = true;
    }

    if (!frontSentence.isEmpty()) {
      if (previousSentenceExists) {
        baseSentence += " and ";
      }
      baseSentence += frontSentence + " in front of you";
      previousSentenceExists = true;
    }

    if (!rightSentence.isEmpty()) {
      if (previousSentenceExists) {
        baseSentence += " and ";
      }
      baseSentence += rightSentence + " on your right";
    }
    return baseSentence;
  }

  private long getCurrentTimeStamp() {
    Date date = new Date();
    return date.getTime();
  }

  private void sayDetectedObjectLocations(List<Classifier.Recognition> results) {
    HashMap<String, ArrayList<String>> objectsToSpeak = getobjectsToSpeak(results);
    String objectsOnLeft = formPartialSentence(objectsToSpeak.get(POSITION_LEFT));
    String objectsInFront = formPartialSentence(objectsToSpeak.get(POSITION_FRONT));
    String objectsOnRight = formPartialSentence(objectsToSpeak.get(POSITION_RIGHT));

    String completeSentence = formSentence(objectsOnLeft, objectsInFront, objectsOnRight);

    // don't speak same thing again unless you've been silent for 5secs
    if (lastSpokenTimeStamp + MIN_STALE_SILENT_DURATION > getCurrentTimeStamp()
      /* && lastSpokenSentence.equals(completeSentence)*/) {
      return;
    }

    lastSpokenTimeStamp = getCurrentTimeStamp();

    Log.d("gonna speak", completeSentence);
    if(canSpeak) {
      if (shouldGuide) {
        tts.speak(completeSentence, TextToSpeech.QUEUE_ADD, null);
      }
      resetObjectsToSpeak();
    } else {
      Log.e("tts error", "cannot speak");
    }
  }

  private String extractDetectedText(FirebaseVisionText result)
  {
    String resultText = result.getText();
    for (FirebaseVisionText.TextBlock block: result.getTextBlocks()) {
      String blockText = block.getText();
      Float blockConfidence = block.getConfidence();
      List<RecognizedLanguage> blockLanguages = block.getRecognizedLanguages();
      Point[] blockCornerPoints = block.getCornerPoints();
      Rect blockFrame = block.getBoundingBox();
      for (FirebaseVisionText.Line line: block.getLines()) {
        String lineText = line.getText();
        Float lineConfidence = line.getConfidence();
        List<RecognizedLanguage> lineLanguages = line.getRecognizedLanguages();
        Point[] lineCornerPoints = line.getCornerPoints();
        Rect lineFrame = line.getBoundingBox();
        for (FirebaseVisionText.Element element: line.getElements()) {
          String elementText = element.getText();
          Float elementConfidence = element.getConfidence();
          List<RecognizedLanguage> elementLanguages = element.getRecognizedLanguages();
          Point[] elementCornerPoints = element.getCornerPoints();
          Rect elementFrame = element.getBoundingBox();
        }
      }
    }
    return resultText;
  }



  private void runTextRecognition() {
    if (runningTextRecognition) {
      return;
    }
    runningTextRecognition = true;
    FirebaseVisionImage image = FirebaseVisionImage.fromBitmap(croppedBitmap);
    FirebaseVisionTextRecognizer detector = FirebaseVision.getInstance()
            .getOnDeviceTextRecognizer();
    Log.d("text recognition", "running");
    Task<FirebaseVisionText> result =
            detector.processImage(image)
                    .addOnSuccessListener(new OnSuccessListener<FirebaseVisionText>() {
                      @Override
                      public void onSuccess(FirebaseVisionText firebaseVisionText) {
                        runningTextRecognition = false;
                        String extractedText = extractDetectedText(firebaseVisionText);
                        // TODO : auto-correct the detected text
                        if (!extractedText.isEmpty()) {
                          detectedText = extractedText;
                          recognitionResults.setText("Detected text:\n" + detectedText);
                        } else {
                          detectedText = "";
                          recognitionResults.setText("");
                        }
                        Log.d("detected text", detectedText);
                      }
                    })
                    .addOnFailureListener(
                            new OnFailureListener() {
                              @Override
                              public void onFailure(@NonNull Exception e) {
                                runningTextRecognition = false;
                                e.printStackTrace();
                                Log.d("detected text", e.toString());
                              }
                            });
  }

  private String extractBarcodeText(List<FirebaseVisionBarcode> barcodes) {
    ArrayList<String> extractedBarcodeTexts = new ArrayList<>();
    for (FirebaseVisionBarcode barcode : barcodes) {
      Rect bounds = barcode.getBoundingBox();
      Point[] corners = barcode.getCornerPoints();

      String extractedBarcodeText = barcode.getRawValue();
      Log.d("detected raw bar code data", extractedBarcodeText);
      int valueType = barcode.getValueType();
      // See API reference for complete list of supported types
      try {
        switch (valueType) {
          case FirebaseVisionBarcode.TYPE_WIFI:
            String ssid = barcode.getWifi().getSsid();
            String password = barcode.getWifi().getPassword();
            int type = barcode.getWifi().getEncryptionType();
            if ( ssid != null && !ssid.isEmpty() && password != null && !password.isEmpty()) {
              extractedBarcodeText = String.format("SSID: %s\nPassword: %s\nType: %d", ssid, password, type);
            }
            break;
          case FirebaseVisionBarcode.TYPE_URL:
            String title = barcode.getUrl().getTitle();
            String url = barcode.getUrl().getUrl();
            if (title != null && !title.isEmpty() && url != null && !url.isEmpty()) {
              extractedBarcodeText = String.format("Title: %s\nURL: %s", title, url);
            }
            break;
        }
      } catch (Exception e) {
        e.printStackTrace();
      }
      extractedBarcodeTexts.add(extractedBarcodeText);
    }
    return String.join("\n----------\n", extractedBarcodeTexts);
  }

  private void runBarRecognition() {
    if (runningBarRecognition) {
      return;
    }
    runningBarRecognition = true;
    FirebaseVisionImage image = FirebaseVisionImage.fromBitmap(croppedBitmap);
    FirebaseVisionBarcodeDetector detector = FirebaseVision.getInstance()
            .getVisionBarcodeDetector();
    Log.d("bar code recognition", "running");
    Task<List<FirebaseVisionBarcode>> result = detector.detectInImage(image)
            .addOnSuccessListener(new OnSuccessListener<List<FirebaseVisionBarcode>>() {
              @Override
              public void onSuccess(List<FirebaseVisionBarcode> barcodes) {
                String barcodeText = extractBarcodeText(barcodes);
                runningBarRecognition = false;
                if (!barcodeText.isEmpty() && !extractedBarcodeText.equals(barcodeText)) {
                  extractedBarcodeText = barcodeText;
                  tts.speak("Barcode detected", TextToSpeech.QUEUE_FLUSH, null);
                  barcodeRecognitionResults.setText("Barcode data:\n" + extractedBarcodeText);
                }
                Log.d("detected bar code data", extractedBarcodeText);
              }
            })
            .addOnFailureListener(new OnFailureListener() {
              @Override
              public void onFailure(@NonNull Exception e) {
                runningBarRecognition = false;
                e.printStackTrace();
                Log.d("detected bar code error", e.toString());
              }
            });


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
            runTextRecognition();
            runBarRecognition();
            final long startTime = SystemClock.uptimeMillis();
            List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);

            results = results.stream().filter(
                    result -> result.getLocation() != null &&result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API
            ).collect(Collectors.toList());
            sayDetectedObjectLocations(results);

            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);


            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              canvas.drawRect(location, paint);

              cropToFrameTransform.mapRect(location);

              result.setLocation(location);
              mappedRecognitions.add(result);
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

  private void speakRecognizedText() {
    if(detectedText.isEmpty()) {
      tts.speak(ERROR_COULDNT_READ, TextToSpeech.QUEUE_ADD, null);
    } else {
      tts.speak(detectedText, TextToSpeech.QUEUE_ADD, null);
    }
  }

  @Override
  public synchronized void onResume() {
    super.onResume();
    initSpeechRecognitionListener();
//    audioManager=(AudioManager)this.getSystemService(Context.AUDIO_SERVICE);
//    streamVolume = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC);
//    audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, 0, AudioManager.FLAG_REMOVE_SOUND_AND_VIBRATE);
  }

  private boolean matches(String regex, String text) {
    Pattern pattern = Pattern.compile(regex);
    Matcher matcher = pattern.matcher(text);
    return matcher.find();
  }

  private void initSpeechRecognitionListener() {
    listener =
            new SpeechRecognitionListener(this, new SpeechRecognitionListener.OnSpeechRecognitionResult() {
              @Override
              public void onSuccess(ArrayList<String> matches) {
                for(String match: matches) {
                  Log.d("SpeechRecognitionListener", match);
                  ;
                  if (matches(REGEX_OCR_TRIGGER_SPEECH, match)) {
                    speakRecognizedText();
                    break;
                  } else if (matches(REGEX_START_GUIDING, match)) {
                    lastSpokenTimeStamp = 0;
                    shouldGuide = true;
                  } else if (matches(REGEX_STOP_GUIDING, match)) {
                    shouldGuide = false;
                  }
                }
              }

              @Override
              public void onError(int error) {

              }
            });
    listener.startListening();
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
