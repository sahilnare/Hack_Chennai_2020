// console.log("from backend");

let active_tab_id = -1;
let current_url = "";

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if(/^https:\/\/www\.youtube/.test(tab.url)) {

    if(changeInfo.status === "complete") {
      if(tabId !== active_tab_id) {
        current_url = tab.url;
        active_tab_id = tabId;
        console.log('Active tab: ', active_tab_id);
        chrome.tabs.executeScript(tabId, {file: './js/youtube.js'}, () => {
          console.log("script injected");
          chrome.tabs.get(tabId, tab => console.log(tab));
        });
      }
    } else {
      console.log("Waiting to finish loading...");
      if(tab.url !== current_url) {
        if(active_tab_id != -1) {
          setTimeout(() => chrome.tabs.sendMessage(active_tab_id, {status: 'loading'}), 2000);
        }
      }
    }
  }

  if(/^https:\/\/twitter/.test(tab.url)) {

    if(changeInfo.status === "complete") {
      if(tabId !== active_tab_id) {
        active_tab_id = tabId;
        console.log('Active tab: ', active_tab_id);
        chrome.tabs.executeScript(tabId, {file: './js/twitter.js'}, () => {
          console.log("script injected");
        });
      }
    } else {
      console.log("Waiting to finish loading...");
      if(active_tab_id != -1) {
        chrome.tabs.sendMessage(active_tab_id, {status: 'loading'});
      }
    }
  }

  if(/^https:\/\/www\.instagram/.test(tab.url)) {

    if(changeInfo.status === "complete") {
      if(tabId !== active_tab_id) {
        active_tab_id = tabId;
        console.log('Active tab: ', active_tab_id);
        chrome.tabs.executeScript(tabId, {file: './js/instagram.js'}, () => {
          console.log("script injected");
        });
      }
    } else {
      console.log("Waiting to finish loading...");
      if(active_tab_id != -1) {
        chrome.tabs.sendMessage(active_tab_id, {status: 'loading'});
      }
    }
  }

  if(/^https:\/\/www\.facebook/.test(tab.url)) {
    chrome.tabs.executeScript(null, {file: './js/facebook.js'}, () => {
      console.log("script injected");
    });
  }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if(request.type === "youtube") {
    fetch('https://ebuzzet.com/api/cyberAlly/model', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({sentence: request.comment, website: 'youtube', email: 'meet@gmail.com', username: request.username})
    }).then(res => res.json()).then(res => {
      chrome.tabs.sendMessage(active_tab_id, {status: null, predictions: res.predictions, commentId: request.commentId});
      // console.log(request.comment);
    });
  }

  if(request.type === "twitter") {
    fetch('https://ebuzzet.com/api/cyberAlly/model', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({sentence: request.comment, website: 'twitter', email: 'meet@gmail.com', username: request.username})
    }).then(res => res.json()).then(res => {
      chrome.tabs.sendMessage(active_tab_id, {status: null, predictions: res.predictions, commentId: request.commentId});
      // console.log(res, request.comment);
    });
  }

  if(request.type === "instagram") {
    fetch('https://ebuzzet.com/api/cyberAlly/nsfw', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({imgUrl: request.imgUrl, username: request.username})
    }).then(res => res.json()).then(res => {
      chrome.tabs.sendMessage(active_tab_id, {status: null, result: res.predictions, imageId: request.imageId});
      console.log(request.username);
    });
  }

  if(request.type === "typeAlert") {
    fetch('https://ebuzzet.com/api/cyberAlly/model', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({sentence: request.comment})
    }).then(res => res.json()).then(res => {
      chrome.tabs.sendMessage(active_tab_id, {status: 'typeAlert', predictions: res.predictions});
      // console.log(request.comment);
    });
  }
});

// chrome.webRequest.onBeforeRequest.addListener((details) => {
//   console.log("Youtube request");
//   console.log(details);
//   return { cancel: true }
// }, {urls: ["*://*.youtube.com/*"]});
