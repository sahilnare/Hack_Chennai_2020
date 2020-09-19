var account = "https://www.youtube.com/channel/UCjfGuoS3aSB6OJ0BlEAMkcQ";

function hello() {
  chrome.tabs.executeScript({
    code: `
      fetch('https://ebuzzet.com/api/cyberAllyData/report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: ${JSON.stringify(account)} });
      alert('User reported');
    `
  });
}

document.getElementById('submit-btn').addEventListener('click', hello);

// let submitBtn = document.getElementById('submit-btn');
//
// submitBtn.addEventListener('click', function() {
//
//   let account = document.getElementById('account').value;
//
//   let accData = {
//     account: account
//   }
//
//   console.log(account);
//
  // fetch('https://ebuzzet.com/api/cyberAllyData/report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(account) })
//
// });
