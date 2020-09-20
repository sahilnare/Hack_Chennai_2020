package com.example.sih2020_4;

class Comments {

    String website, comment , username , commentlink , user_id;

    public Comments(String website, String comment ,String username , String commentlink , String user_id ) {
        this.website= website;
        this.comment = comment;
        this.username = username;
        this.commentlink= commentlink;
        this.user_id = user_id;
    }



    public String getWebsite() {
        return website;
    }

    public String getComment() {
        return comment;
    }

    public String getCommentlink() {
        return commentlink;
    }

    public String getUser_id() {
        return user_id;
    }

    public String getUsername() {
        return username;
    }
}
