package org.asmita.objectdetection;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.util.Log;

import java.util.ArrayList;

public class SpeechRecognitionListener implements RecognitionListener {
    private String TAG = "SpeechRecognitionListener";

    public interface OnSpeechRecognitionResult {
        void onSuccess(ArrayList<String> matches);
        void onError(int error);
    }

    private SpeechRecognizer mSpeechRecognizer;
    private Intent mSpeechRecognizerIntent;
    private OnSpeechRecognitionResult onSpeechRecognitionResult;
    private Context context;

    public SpeechRecognitionListener(Context context, OnSpeechRecognitionResult speechRecognitionResult) {
        this.context = context;
        onSpeechRecognitionResult = speechRecognitionResult;
        init();
    }

    private void init() {
        destroy();
        mSpeechRecognizer = SpeechRecognizer.createSpeechRecognizer(context);
        mSpeechRecognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        mSpeechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        mSpeechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE,
                context.getPackageName());
        mSpeechRecognizer.setRecognitionListener(this);
    }

    public void startListening() {
        mSpeechRecognizer.startListening(mSpeechRecognizerIntent);
    }

    public void stopListening() {
        mSpeechRecognizer.stopListening();
    }

    public void destroy() {
        if (mSpeechRecognizer != null) {
            mSpeechRecognizer.destroy();
        }
    }

    @Override
    public void onBeginningOfSpeech()
    {
        Log.d(TAG, "onBeginingOfSpeech");
    }

    @Override
    public void onBufferReceived(byte[] buffer)
    {

    }

    @Override
    public void onEndOfSpeech()
    {
        Log.d(TAG, "onEndOfSpeech");
    }

    @Override
    public void onError(int error)
    {
        onSpeechRecognitionResult.onError(error);
        Log.d(TAG, "error = " + error);
        init();
        startListening();
    }

    @Override
    public void onEvent(int eventType, Bundle params)
    {

    }

    @Override
    public void onPartialResults(Bundle partialResults)
    {

    }

    @Override
    public void onReadyForSpeech(Bundle params)
    {
        Log.d(TAG, "onReadyForSpeech"); //$NON-NLS-1$
    }

    @Override
    public void onResults(Bundle results)
    {
        ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
        Log.d(TAG, matches.size()+"");
        for (String match: matches) {
            Log.d(TAG, match);
        }
        onSpeechRecognitionResult.onSuccess(matches);
        startListening();
    }

    @Override
    public void onRmsChanged(float rmsdB)
    {
    }
}