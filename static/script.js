// Basic animation with GSAP
gsap.from(".logo", { duration: 1, x: -100, opacity: 0 });
gsap.from(".profile-icon", { duration: 1, x: 100, opacity: 0 });
gsap.from(".hero h1", { duration: 1.2, y: -50, opacity: 0 });
gsap.from(".hero p", { duration: 1.4, y: 50, opacity: 0 });

function toggleProfile() {
  alert("Redirecting to login/profile page...");
}
function handleLogin() {
    const loginBtn = document.getElementById('loginBtn');
    const profileIcon = document.getElementById('profileIcon');
  
    loginBtn.style.display = 'none';
    profileIcon.style.display = 'block';
    alert("You are now logged in!");
  }

  function toggleDropdown() {
    const menu = document.getElementById('dropdownMenu');
    menu.style.display = menu.style.display === 'block' ? 'none' : 'block';
  }
  
  function handleLogout() {
    const loginBtn = document.getElementById('loginBtn');
    const profileWrapper = document.querySelector('.profile-wrapper');
    const dropdownMenu = document.getElementById('dropdownMenu');
  
    loginBtn.style.display = 'inline-block';
    profileWrapper.style.display = 'none';
    dropdownMenu.style.display = 'none';
  }  