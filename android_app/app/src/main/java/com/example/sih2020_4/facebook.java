package com.example.sih2020_4;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
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

public class facebook extends AppCompatActivity {

    private static final String JSON_URL = "https://ebuzzet.com/api/cyberAllyData/allcomments/facebook";

    //listview object
    ListView listView;

    //the hero list where we will store all the hero objects after parsing json
    List<Comments> CommentList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_facebook);

        Log.d("bubby", "you made it to facebookssssss");

        //initializing listview and hero list
        listView = (ListView) findViewById(R.id.listView);
        CommentList = new ArrayList<>();

        //this method will fetch and parse the data
        loadCommentList();
    }

    private void loadCommentList() {
        //getting the progressbar
        final ProgressBar progressBar = (ProgressBar) findViewById(R.id.progressBar);

        //making the progressbar visible
        progressBar.setVisibility(View.VISIBLE);

        //creating a string request to send request to the url
        StringRequest stringRequest = new StringRequest(Request.Method.POST, JSON_URL,
                new Response.Listener<String>() {
                    @Override
                    public void onResponse(String response) {
                        //hiding the progressbar after completion
                        progressBar.setVisibility(View.INVISIBLE);


                        try {
                            //getting the whole json object from the response
                            JSONArray commentArray = new JSONArray(response);

                            //now looping through all the elements of the json array
                            for (int i = 0; i < commentArray.length(); i++) {
                                //getting the json object of the particular index inside the array
                                JSONObject jsonObject = commentArray.getJSONObject(i);

                                //creating a hero object and giving them the values from json object
                                Comments comment = new Comments(jsonObject.getString("website"), jsonObject.getString("comment"),jsonObject.getString("username"),jsonObject.getString("commentlink"),jsonObject.getString("user_id"));

                                //adding the hero to Commentlist
                                CommentList.add(comment);
                            }

                            //creating custom adapter object
                            ListViewAdapter adapter = new ListViewAdapter(CommentList, getApplicationContext());

                            //adding the adapter to listview
                            listView.setAdapter(adapter);

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
                }){
            @Override
            public byte[] getBody() throws AuthFailureError {
                HashMap<String, String> params2 = new HashMap<String, String>();
                String email;
                Intent intent = getIntent();
                email = intent.getStringExtra("email");
                params2.put("email", email);

                return new JSONObject(params2).toString().getBytes();
            }

            @Override
            public String getBodyContentType() {
                return "application/json";
            }
        };

        //creating a request queue
        RequestQueue requestQueue = Volley.newRequestQueue(this);

        //adding the string request to request queue
        requestQueue.add(stringRequest);
    }

}
