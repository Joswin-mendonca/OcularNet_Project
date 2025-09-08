window.onload = function() {
  const dataStr = localStorage.getItem('patientData');
  if (!dataStr) {
    document.getElementById('result').innerHTML = '<p>No data found. Please enter information first.</p>';
    return;
  }
  const data = JSON.parse(dataStr);
  const resultDiv = document.getElementById('result');
  const imgHtml = `<img src="${data.imageBase64}" alt="Uploaded Image" style="max-width:100%; border-radius:8px;"/>`;
  resultDiv.innerHTML = `
    <p><strong>Patient Name:</strong> ${data.patientName}</p>
    <p><strong>Age:</strong> ${data.age}</p>
    <p><strong>Sex:</strong> ${data.sex}</p>
    <p><strong>Phone Number:</strong> ${data.phone}</p>
    <p><strong>Date of Birth:</strong> ${data.dob}</p>
    ${imgHtml}
  `;
};

document.getElementById('backBtn').addEventListener('click', function() {
  localStorage.removeItem('patientData');
  window.location.href = 'index.html';
});
