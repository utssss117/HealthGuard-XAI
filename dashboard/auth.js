/**
 * auth.js
 * Handle login, registration, and token management for HealthGuard-XAI
 */

const AUTH_API = "http://localhost:8000/auth";

const TOKEN_KEY = "healthguard_jwt";
const USER_KEY = "healthguard_user";

// --- Token Management ---

function setAuth(tokenData) {
  localStorage.setItem(TOKEN_KEY, tokenData.access_token);
  localStorage.setItem(USER_KEY, JSON.stringify({
    role: tokenData.role,
    full_name: tokenData.full_name
  }));
}

function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function getUser() {
  const u = localStorage.getItem(USER_KEY);
  return u ? JSON.parse(u) : null;
}

function isAuthenticated() {
  return !!getToken();
}

/** Returns the Authorization header for fetch requests */
function authHeaders() {
  const token = getToken();
  return token ? { "Authorization": `Bearer ${token}` } : {};
}

function logout() {
  clearAuth();
  window.location.href = "login.html";
}


// --- Login Page Logic (only runs if we are on login.html) ---

if (window.location.pathname.endsWith("login.html")) {
  
  // Auto-redirect if already logged in
  if (isAuthenticated()) {
    window.location.href = "index.html";
  }

  const loginForm = document.getElementById("login-form");
  const regForm = document.getElementById("register-form");
  const alertBox = document.getElementById("alert-box");

  function showAlert(msg, type = "error") {
    alertBox.textContent = msg;
    alertBox.className = `alert ${type}`;
  }

  function toggleBtnLoad(btn, loading) {
    const text = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.btn-loader');
    if (loading) {
      text.classList.add('hidden');
      loader.classList.remove('hidden');
      btn.disabled = true;
    } else {
      text.classList.remove('hidden');
      loader.classList.add('hidden');
      btn.disabled = false;
    }
  }

  // Handle Login
  if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = document.getElementById("login-btn");
      toggleBtnLoad(btn, true);
      alertBox.classList.add("hidden");

      const email = document.getElementById("login-email").value;
      const password = document.getElementById("login-password").value;

      try {
        const res = await fetch(`${AUTH_API}/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password })
        });
        
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Login failed");

        setAuth(data);
        window.location.href = "index.html"; // Redirect to dashboard
      } catch (err) {
        showAlert(err.message);
      } finally {
        toggleBtnLoad(btn, false);
      }
    });
  }

  // Handle Registration
  if (regForm) {
    regForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = document.getElementById("register-btn");
      toggleBtnLoad(btn, true);
      alertBox.classList.add("hidden");

      const payload = {
        full_name: document.getElementById("reg-name").value,
        email: document.getElementById("reg-email").value,
        password: document.getElementById("reg-password").value,
        role: document.getElementById("reg-role").value,
        hospital_id: document.getElementById("reg-hospital").value
      };

      try {
        const res = await fetch(`${AUTH_API}/register`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Registration failed");

        setAuth(data);
        window.location.href = "index.html"; // Redirect to dashboard
      } catch (err) {
        showAlert(err.message);
      } finally {
        toggleBtnLoad(btn, false);
      }
    });
  }
}
