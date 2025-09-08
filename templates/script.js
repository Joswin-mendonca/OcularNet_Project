const loginPage = document.getElementById('loginPage');
const mainPage = document.getElementById('mainPage');
const detectPage = document.getElementById('detectPage');
const messageDiv = document.getElementById('message');

const USERNAME = 'OcularNet';
const PASSWORD = '4mt22ci';

document.getElementById('loginBtn').addEventListener('click', function () {
  const username = document.getElementById('username').value.trim();
  const password = document.getElementById('password').value.trim();
  if (username === USERNAME && password === PASSWORD) {
    messageDiv.textContent = '';
    loginPage.style.display = 'none';
    mainPage.style.display = 'block';
  } else {
    messageDiv.textContent = 'Incorrect username or password';
  }
});

document.getElementById('detectBtn').addEventListener('click', function () {
  mainPage.style.display = 'none';
  detectPage.style.display = 'block';
});

const patientForm = document.getElementById('patientForm');
const imageUpload = document.getElementById('imageUpload');

patientForm.addEventListener('submit', function (e) {
  e.preventDefault();

  const name = document.getElementById('patientName').value;
  const age = document.getElementById('age').value;
  const sex = document.getElementById('sex').value;
  const phone = document.getElementById('phone').value;
  const dob = document.getElementById('dob').value;
  const file = imageUpload.files[0];

  if (!file) {
    alert('Please upload an image.');
    return;
  }

  const reader = new FileReader();
  reader.onload = function(event) {
    const formData = {
      patientName: name,
      age: age,
      sex: sex,
      phone: phone,
      dob: dob,
      imageBase64: event.target.result
    };
    localStorage.setItem('patientData', JSON.stringify(formData));

    window.location.href = 'result.html';
  };
  reader.readAsDataURL(file);
});
