package com.example.sih_keyboard;


import android.graphics.Color;
import android.inputmethodservice.InputMethodService;
import android.inputmethodservice.Keyboard;
import android.inputmethodservice.KeyboardView;
import android.media.AudioManager;
import android.util.Log;
import android.view.View;
import android.view.inputmethod.ExtractedText;
import android.view.inputmethod.ExtractedTextRequest;
import android.view.inputmethod.InputConnection;
import android.widget.Toast;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.HashMap;


public class SMART_KEYBOARD extends InputMethodService implements KeyboardView.OnKeyboardActionListener {

    private KeyboardView kv;
    private Keyboard keyboard;
    private boolean isCaps = false;
    ExtractedText text;
    InputConnection ic;
    private static final String JSON_URL = "https://ebuzzet.com/api/cyberAlly/model";

    @Override
    public View onCreateInputView() {
        kv = (KeyboardView) getLayoutInflater().inflate(R.layout.keyboard, null);
        keyboard = new Keyboard(this, R.xml.qwerty);
        kv.setKeyboard(keyboard);
        kv.setOnKeyboardActionListener(this);
        ic = getCurrentInputConnection();
        return kv;
    }

    @Override
    public void onPress(int i) {
        switch (i) {
            case Keyboard.KEYCODE_DONE:
                Log.i("entered","you are inside onpress");
                text = ic.getExtractedText(new ExtractedTextRequest(), 1);
                Log.i("onpress", String.valueOf(text.text));



                StringRequest stringRequest = new StringRequest(Request.Method.POST, JSON_URL,
                        new Response.Listener<String>() {
                            @Override
                            public void onResponse(String response) {
                                Integer flag = 0 ;
                                JSONObject serverresult ;
                                try {
                                    serverresult = new JSONObject(response);
                                    JSONArray resultarray = serverresult.getJSONArray("predictions");
                                    for (int i = 0; i < 7; i++) {
                                       JSONObject labelobj = resultarray.getJSONObject(i);
                                       JSONArray result_array = labelobj.getJSONArray("results");
                                       JSONObject obj_inside_result = result_array.getJSONObject(0);
                                       String match = obj_inside_result.getString("match");
                                       if(match.contains("true")){
                                           flag = 1;
                                       }


                                        Log.i("server", String.valueOf(flag));
                                    }
                                    if(flag == 1){
                                        Log.i("server", String.valueOf("message is toxic"));
//                                        CharSequence beforCursorText = ic.getTextBeforeCursor(text.text.length(), 0);
//                                        CharSequence afterCursorText = ic.getTextAfterCursor(text.text.length(), 0);
//                                        ic.deleteSurroundingText(beforCursorText.length(), afterCursorText.length());
//                                        Toast.makeText(getApplicationContext(), "MESSAGE IS TOXIC", Toast.LENGTH_LONG).show();
                                        Toast toast = Toast.makeText(getApplicationContext(), "MESSAGE IS TOXIC", Toast.LENGTH_LONG);
                                        toast.getView().setBackgroundColor(Color.parseColor("#ff0000"));
                                        toast.show();


                                    }else{

                                        Toast.makeText(getApplicationContext(), "MESSAGE IS INNOCENT", Toast.LENGTH_LONG).show();
                                    }


                                } catch (JSONException e) {
                                    e.printStackTrace();
                                }

                            }
                        },
                        new Response.ErrorListener() {
                            @Override
                            public void onErrorResponse(VolleyError error) {
                                //displaying the error in toast if occurrs
                                Toast.makeText(getApplicationContext(), "no no no no", Toast.LENGTH_SHORT).show();
                            }
                        })
                {
                    @Override
                    public byte[] getBody() throws AuthFailureError {
                        HashMap<String, String> params2 = new HashMap<String, String>();

                        params2.put("sentence", String.valueOf(text.text));
                        params2.put("website", String.valueOf(""));
                        params2.put("username", String.valueOf(""));
                        params2.put("email", String.valueOf(""));

                        return new JSONObject(params2).toString().getBytes();
                    }

                    @Override
                    public String getBodyContentType()
                    {
                        return "application/json";
                    }
                };

                //creating a request queue
                RequestQueue requestQueue = Volley.newRequestQueue(this);

                //adding the string request to request queue
                requestQueue.add(stringRequest);

                break;

        }


    }



    @Override
    public void onRelease(int i) {

    }

    @Override
    public void onKey(int i, int[] ints) {


        playClick(i);
        switch (i) {
            case Keyboard.KEYCODE_DELETE:
                ic.deleteSurroundingText(1, 0);
                break;
            case Keyboard.KEYCODE_SHIFT:
                isCaps = !isCaps;
                keyboard.setShifted(isCaps);
                kv.invalidateAllKeys();
                break;
            case Keyboard.KEYCODE_DONE:
                ic.commitText("", 1);
//                text = (ExtractedText) ic.getExtractedText(new ExtractedTextRequest(),1);
//                Log.i("onkey", String.valueOf(text.text));
//                ic.sendKeyEvent(new KeyEvent(KeyEvent.ACTION_DOWN,KeyEvent.KEYCODE_ENTER));
                break;
            default:
                char code = (char) i;
                if (Character.isLetter(code) && isCaps)
                    code = Character.toUpperCase(code);
                ic.commitText(String.valueOf(code), 1);


        }


    }

    private void playClick(int i) {

        AudioManager am = (AudioManager) getSystemService(AUDIO_SERVICE);
        switch (i) {
            case 32:
                am.playSoundEffect(AudioManager.FX_KEYPRESS_SPACEBAR);
                break;
            case Keyboard.KEYCODE_DONE:
            case 10:
                am.playSoundEffect(AudioManager.FX_KEYPRESS_RETURN);
                break;
            case Keyboard.KEYCODE_DELETE:
                am.playSoundEffect(AudioManager.FX_KEYPRESS_DELETE);
                break;
            default:
                am.playSoundEffect(AudioManager.FX_KEYPRESS_STANDARD);


        }
    }

    @Override
    public void onText(CharSequence charSequence) {

    }

    @Override
    public void swipeLeft() {

    }

    @Override
    public void swipeRight() {

    }

    @Override
    public void swipeDown() {

    }

    @Override
    public void swipeUp() {

    }
}
