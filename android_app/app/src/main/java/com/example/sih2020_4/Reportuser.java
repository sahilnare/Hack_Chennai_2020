package com.example.sih2020_4;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.ProgressBar;
import android.widget.TextView;
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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Reportuser extends AppCompatActivity {

    String user;

    private static final String REPORT_URL = "https://ebuzzet.com/api/cyberAllyData/report";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.reportuser);


        Button report = findViewById(R.id.reportbtn);
        report.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View view) {
            EditText usermail = (EditText) findViewById(R.id.userreport);
            user = usermail.getEditableText().toString();

            final ProgressBar progressBar = (ProgressBar) findViewById(R.id.progressBar);
            progressBar.setVisibility(View.VISIBLE);
            StringRequest stringRequest = new StringRequest(Request.Method.POST, REPORT_URL,
                    new Response.Listener<String>() {
                        @Override
                        public void onResponse(String response) {
                            //hiding the progressbar after completion
                            progressBar.setVisibility(View.INVISIBLE);

                            if (response.contains("Success")){
                                Log.d("in case of success", (response));
                                Toast.makeText(getApplicationContext(),"USER ID REPORTED", Toast.LENGTH_LONG).show();

                            }else {
                                Log.d("in case of failure", (response));
                            }
                        }
                    },
                    new Response.ErrorListener() {
                        @Override
                        public void onErrorResponse(VolleyError error) {
                            //displaying the error in toast if occurs
                            progressBar.setVisibility(View.INVISIBLE);
                            Toast.makeText(getApplicationContext(),"NOT REPORTED", Toast.LENGTH_LONG).show();
                        }
                    }) {
                @Override
                public byte[] getBody() throws AuthFailureError {
                    HashMap<String, String> params2 = new HashMap<String, String>();
                    params2.put("account", user);

                    return new JSONObject(params2).toString().getBytes();
                }

                @Override
                public String getBodyContentType() {
                    return "application/json";
                }
            };

            RequestQueue requestQueue =  Volley.newRequestQueue(getApplicationContext());

            //adding the string request to request queue
            requestQueue.add(stringRequest);
        }
    });



}
}

