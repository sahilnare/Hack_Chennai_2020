//document.querySelectorAll(".css-901oao.r-jwli3a.r-1qd0xha.r-a023e6.r-16dba41.r-ad9z0x.r-bcqeeo.r-bnwqim.r-qvutc0").forEach(item => console.log(item.innerText))

var commentArray = {};
var commentId = 0;
var matched = false;
var commentLoad;
var username = '';
var totalTweetsLoaded = 3;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if(request.status === 'loading') {
    totalTweetsLoaded = 5;
    document.querySelectorAll('.css-901oao.r-1qd0xha.r-a023e6.r-16dba41.r-ad9z0x.r-bcqeeo.r-bnwqim.r-qvutc0').forEach((item, i) => {
      item.classList.remove("exclude");
    });
  }
  else {
    console.log(request.predictions, commentArray[request.commentId].innerText);
    for(i=0; i<7; i++){
      if(request.predictions[i].results[0].match){
          console.log(request.predictions[i].label + " was found with probability of " + request.predictions[i].results[0].probabilities[1]);
          matched = true;
      }
    }
    if(matched) {document.querySelectorAll(".css-901oao.r-1qd0xha.r-a023e6.r-16dba41.r-ad9z0x.r-bcqeeo.r-bnwqim.r-qvutc0")
      // console.log(commentArray[request.commentId].innerText);
      commentArray[request.commentId].innerText = 'This comment is inappropriate.';
      commentArray[request.commentId].style.color = 'red';
    }
    // if(request.result > 0.8) {
    //   commentArray[request.commentId].innerText = 'This comment is inappropriate.';
    //   commentArray[request.commentId].style.color = 'red';
    // }
    matched = false;
  }
});
// Scrap comments, tweets and send to background.js

function sendCommentsToBackground() {
  totalTweetsLoaded = document.querySelectorAll(".css-901oao.r-1qd0xha.r-a023e6.r-16dba41.r-ad9z0x.r-bcqeeo.r-bnwqim.r-qvutc0").length;
  document.querySelectorAll(".css-901oao.r-1qd0xha.r-a023e6.r-16dba41.r-ad9z0x.r-bcqeeo.r-bnwqim.r-qvutc0:not(.exclude)").forEach(item => {
    commentId = "ID" + Math.round(Math.random()*10000);
    username = Array.from(item.parentElement.parentElement.parentElement.querySelectorAll('span')).filter(item => item.textContent.includes('@') && item.classList.contains('css-901oao'))[0].textContent;
    chrome.runtime.sendMessage({type: "twitter", comment: item.innerText, commentId: commentId, username: username});
    commentArray[commentId] = item;
    item.classList.add("exclude");
  });
  commentCheckTimer();
}

function commentCheck() {
  if(document.querySelectorAll(".css-901oao.r-1qd0xha.r-a023e6.r-16dba41.r-ad9z0x.r-bcqeeo.r-bnwqim.r-qvutc0").length <= totalTweetsLoaded) {
    console.log("Tweets loading...");
  } else {
    console.log("Tweets loaded");
    clearInterval(commentLoad);
    setTimeout(() => sendCommentsToBackground(), 3000);
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
