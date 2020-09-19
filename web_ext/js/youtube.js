// console.log("from frontend");

var commentArray = {};
var commentId = 0;
var matched = false;
var matched2 = false;
var commentLoad;
var totalCommentsLoaded = 6;
var executed = false;
var username = '';
var wordCheck;
var typedCom;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if(request.status === 'loading') {

    totalCommentsLoaded = 6;

    document.querySelectorAll('#content-text').forEach((item, i) => {
      item.classList.remove("exclude");
      item.style.visibility = "visible";
      item.parentElement.style.border = "none";
    });
  }
  else if(request.status === 'typeAlert') {
    console.log(request.predictions);
    for(i=0; i<7; i++){
      if(request.predictions[i].results[0].match){
          // console.log(request.predictions[i].label + " was found with probability of " + request.predictions[i].results[0].probabilities[1]);
          matched2 = true;
      }
    }
    if(matched2) {
      document.getElementById("input-container").style.border = "thick solid #FF0000";
    } else {
      document.getElementById("input-container").style.border = "none";
    }
    matched2 = false;
  }
  else {
    console.log(request.predictions, commentArray[request.commentId].innerText);
    for(i=0; i<7; i++){
      if(request.predictions[i].results[0].match){
          console.log(request.predictions[i].label + " was found with probability of " + request.predictions[i].results[0].probabilities[1]);
          matched = true;
      }
    }
    if(matched) {
      // console.log(commentArray[request.commentId].innerText);
      // commentArray[request.commentId].textContent = 'This comment is inappropriate.';
      // commentArray[request.commentId].style.color = 'red';
      commentArray[request.commentId].parentElement.style.border = "thick solid #FF0000";
      commentArray[request.commentId].style.visibility = "hidden";
    }
    matched = false;
  }
});
// Scrap comments, tweets and send to background.js

function sendCommentsToBackground() {
  if(!executed) {
    // clearInterval(wordCheck);
    executed = true;
    inputListFunc();
  }

  totalCommentsLoaded = document.querySelectorAll('#content-text').length;
  document.querySelectorAll('#content-text:not(.exclude)').forEach((item, i) => {
    commentId = "ID" + Math.round(Math.random()*10000);
    username = item.parentElement.parentElement.parentElement.querySelector('#header').querySelector('#author-text').querySelector('span').textContent.trim();
    chrome.runtime.sendMessage({type: "youtube", comment: item.innerText, commentId: commentId, username: username});
    commentArray[commentId] = item;
    item.classList.add("exclude");
    // console.log(item.innerText);
  });
  commentCheckTimer();
}

function commentCheck() {
  if(document.querySelectorAll('#content-text').length <= totalCommentsLoaded) {
    console.log("Comments loading...");
  } else {
    // console.log(totalCommentsLoaded);
    console.log("Comments loaded");
    clearInterval(commentLoad);
    setTimeout(() => sendCommentsToBackground(), 2500);
  }
}

function commentCheckTimer() {
  commentLoad = setInterval( function() {
    commentCheck();
  }, 2000);
}

if(commentLoad) {
  clearInterval(commentLoad);
  commentCheckTimer();
} else {
  commentCheckTimer();
}

function inputListFunc() {
  setTimeout(() => typeAlert(), 500);
}

function typeAlert() {
  console.log('TypeAlert activated');

  wordCheck = setInterval( function() {

    if(document.getElementById("contenteditable-root")) {

      typedCom = document.getElementById("contenteditable-root").textContent;
      if(typedCom.length > 0) {
        // console.log(typedCom);
        chrome.runtime.sendMessage({type: "typeAlert", comment: typedCom});
      } else {
        document.getElementById("input-container").style.border = "none";
      }
    }
  }, 2000);
}
