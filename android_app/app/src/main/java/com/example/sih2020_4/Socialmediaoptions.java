package com.example.sih2020_4;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

public class Socialmediaoptions extends AppCompatActivity {
    String email;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_socialmediaoptions);


        Intent intent = getIntent();
        email = intent.getStringExtra("email");

        TextView fb = (TextView) findViewById(R.id.fbtextview);
        fb.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Socialmediaoptions.this, facebook.class);
                intent.putExtra("email", email);
                startActivity(intent);

            }

        });

        TextView youtube = (TextView) findViewById(R.id.youtubetextview);
        youtube.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Socialmediaoptions.this, Youtube.class);
                intent.putExtra("email", email);
                startActivity(intent);

            }

        });

        TextView twitter = (TextView) findViewById(R.id.twittertextview);
        twitter.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Socialmediaoptions.this, Twitter.class);
                intent.putExtra("email", email);
                startActivity(intent);

            }

        });

        TextView insta = (TextView) findViewById(R.id.reportview);
        insta.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Socialmediaoptions.this, Reportuser.class);
                intent.putExtra("email", email);
                startActivity(intent);

            }

        });

    }
}
