console.log("from frontend");

// let class = "oi732d6d ik7dh3pa d2edcug0 qv66sw1b c1et5uql a8c37x1j muag1w35 enqfppq2 jq4qci2q a3bd9o3v knj5qynh oo9gr5id"
// let child = "kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x c1et5uql"

// Scrap comments, tweets and send to background.js

document.querySelectorAll('.oi732d6d.ik7dh3pa.d2edcug0.qv66sw1b.c1et5uql.a8c37x1j.muag1w35.enqfppq2.jq4qci2q.a3bd9o3v.knj5qynh.oo9gr5id .cxmmr5t8').forEach((item, i) => {
  // chrome.runtime.sendMessage({comment: item.innerText});
  if(item.children[0].innerText === undefined) {
    console.log(item.children[0].innerText);
  }
});
