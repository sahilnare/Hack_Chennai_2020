package com.example.sih2020_4;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import java.util.List;

public class ListViewAdapter extends ArrayAdapter<Comments> {

    //the hero list that will be displayed
    private List<Comments> CommentList;

    //the context object
    private Context mCtx;

    //here we are getting the commentlist and context
    //so while creating the object of this adapter class we need to give herolist and context
    public ListViewAdapter(List<Comments> CommentList, Context mCtx) {
        super(mCtx, R.layout.list_items, CommentList);
        this.CommentList = CommentList;
        this.mCtx = mCtx;
    }

    //this method will return the list item
    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        //getting the layoutinflater
        LayoutInflater inflater = LayoutInflater.from(mCtx);

        //creating a view with our xml layout
        View listViewItem = inflater.inflate(R.layout.list_items, null, true);

        //getting text views
        TextView textViewwebsite = listViewItem.findViewById(R.id.textViewwebsite);
        TextView textViewcomment = listViewItem.findViewById(R.id.textViewcomment);
        TextView textViewcommentlink = listViewItem.findViewById(R.id.textViewcommentlink);
        TextView textViewusername = listViewItem.findViewById(R.id.textViewusername);
        TextView textViewuser_id = listViewItem.findViewById(R.id.textViewuser_id);

        //Getting the hero for the specified position
        Comments comments = CommentList.get(position);

        //setting hero values to textviews
        textViewwebsite.setText(comments.getWebsite());
        textViewcomment.setText(comments.getComment());
        textViewcommentlink.setText(comments.getCommentlink());
        textViewusername.setText(comments.getUsername());
        textViewuser_id.setText(comments.getUser_id());

        //returning the listitem
        return listViewItem;
    }
}
