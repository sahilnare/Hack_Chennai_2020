// console.log("from frontend");

var imageArray = {};
var imageId = 0;
var matched = false;
var imageLoad;
var totalImagesLoaded = 5;
var username = '';

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.status === 'loading') {
    totalImagesLoaded = 5;
    // document.querySelectorAll('.KL4Bh > .FFVAD').forEach((item, i) => {
    //   item.classList.remove("exclude");
    // });
    console.log("reloaded");
  }
  else {
    console.log(request.result, imageArray[request.imageId]);
    request.result.forEach(item => {
      if(item.className === "Sexy") {
        if(item.probability > 0.8) {
          //console.log(request.result, imageArray[request.imageId]);
          var coverImg = document.createElement("IMG");
          coverImg.setAttribute("src", "https://instagram.fbom5-1.fna.fbcdn.net/v/t51.2885-15/sh0.08/e35/s640x640/119775731_373355300498013_8508589766852102418_n.jpg?_nc_ht=instagram.fbom5-1.fna.fbcdn.net&_nc_cat=109&_nc_ohc=DynOQ752mdwAX-e-DS-&oh=fbcb955615c0947cbbf3e923a4ec5100&oe=5F8C2172");
          coverImg.setAttribute('style', 'z-index:1000');
          coverImg.classList.add('FFVAD');
          imageArray[request.imageId].parentElement.appendChild(coverImg);
        }
      }
      if(item.className === "Porn") {
        if(item.probability > 0.6) {
          //console.log(request.result, imageArray[request.imageId]);
          var coverImg = document.createElement("IMG");
          coverImg.setAttribute("src", "https://instagram.fbom5-1.fna.fbcdn.net/v/t51.2885-15/sh0.08/e35/s640x640/119775731_373355300498013_8508589766852102418_n.jpg?_nc_ht=instagram.fbom5-1.fna.fbcdn.net&_nc_cat=109&_nc_ohc=DynOQ752mdwAX-e-DS-&oh=fbcb955615c0947cbbf3e923a4ec5100&oe=5F8C2172");
          coverImg.setAttribute('style', 'z-index:1000');
          coverImg.classList.add('FFVAD');
          imageArray[request.imageId].parentElement.appendChild(coverImg);
        }
      }
    });
  }

});
// Scrap comments, tweets and send to background.js

function sendImagesToBackground() {
  totalImagesLoaded = document.querySelectorAll(".KL4Bh > .FFVAD").length;
  document.querySelectorAll(".KL4Bh > .FFVAD:not(.exclude)").forEach((item, i) => {
    imageId = "ID" + Math.round(Math.random()*10000);
    if(item.parentElement.parentElement.parentElement.parentElement.parentElement.querySelector('.sqdOP.yWX7d._8A5w5.ZIAjV')) {
      username = item.parentElement.parentElement.parentElement.parentElement.parentElement.querySelector('.sqdOP.yWX7d._8A5w5.ZIAjV').href;
    } else if(document.querySelector("._7UhW9.fKFbl.yUEEX.KV-D4.fDxYl")) {
      username = document.querySelector("._7UhW9.fKFbl.yUEEX.KV-D4.fDxYl").textContent;
    }
    else {
      username = 'none';
    }
    chrome.runtime.sendMessage({type: "instagram", imgUrl: item.src, imageId: imageId, username: username});
    imageArray[imageId] = item;
    // console.log(item.src, imageId);
    item.classList.add("exclude");
  });
  imageCheckTimer();
}

function imageCheck() {
  if(document.querySelectorAll(".KL4Bh > .FFVAD").length <= totalImagesLoaded) {
    console.log("Images loading...");
  } else {
    console.log("Images loaded");
    clearInterval(imageLoad);
    setTimeout(() => sendImagesToBackground(), 3000);
  }
}

function imageCheckTimer() {
  imageLoad = setInterval( function() {
    imageCheck();
  }, 2000);
}

if(imageLoad) {
  clearInterval(imageLoad);
  imageCheckTimer();
} else {
  imageCheckTimer();
}


//item.src = "https://instagram.fbom5-1.fna.fbcdn.net/v/t51.2885-15/sh0.08/e35/s750x750/110268338_232032427802843_3475931746761148163_n.jpg?_nc_ht=instagram.fbom5-1.fna.fbcdn.net&_nc_cat=111&_nc_ohc=16A1aPjfasIAX9P1dN7&oh=dfed590d7a05cf4e09814876604a7b23&oe=5F4B44BA";
//item.srcset="https://instagram.fbom5-1.fna.fbcdn.net/v/t51.2885-15/sh0.08/e35/s640x640/110268338_232032427802843_3475931746761148163_n.jpg?_nc_ht=instagram.fbom5-1.fna.fbcdn.net&amp;_nc_cat=111&amp;_nc_ohc=16A1aPjfasIAX9P1dN7&amp;oh=53742ad19d2607275c4fa3ebad3984f7&amp;oe=5F4AD2FE 640w,https://instagram.fbom5-1.fna.fbcdn.net/v/t51.2885-15/sh0.08/e35/s750x750/110268338_232032427802843_3475931746761148163_n.jpg?_nc_ht=instagram.fbom5-1.fna.fbcdn.net&amp;_nc_cat=111&amp;_nc_ohc=16A1aPjfasIAX9P1dN7&amp;oh=dfed590d7a05cf4e09814876604a7b23&amp;oe=5F4B44BA 750w,https://instagram.fbom5-1.fna.fbcdn.net/v/t51.2885-15/e35/110268338_232032427802843_3475931746761148163_n.jpg?_nc_ht=instagram.fbom5-1.fna.fbcdn.net&amp;_nc_cat=111&amp;_nc_ohc=16A1aPjfasIAX9P1dN7&amp;oh=e2a4da2aead778d74928cec5a386ca8b&amp;oe=5F4C8EC8 1080w";
