// ================================================================
//  NeuroScan AI — MongoDB API Client
//  Drop-in replacement for firebase.js (Firestore functions)
//
//  Auth:    Still uses Firebase Auth (Google OAuth + Email/Password)
//  Storage: All data goes through Express + MongoDB backend API
//
//  All functions keep the SAME signatures as the old firebase.js
//  so no page code needs to change — only this import path changes.
//
//  SECURITY: Firebase config is fetched from the backend /api/config
//  endpoint at runtime, so credentials are never hardcoded in source.
//  Set FIREBASE_API_KEY, FIREBASE_AUTH_DOMAIN, etc. in your .env file.
// ================================================================

import { initializeApp }        from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js';
import { getAuth, GoogleAuthProvider, signInWithPopup,
         signInWithEmailAndPassword, createUserWithEmailAndPassword,
         signOut, onAuthStateChanged, updateProfile }
  from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js';

// ── CONFIGURATION — loaded from server at runtime ─────────────
// Firebase config values are injected by the backend /api/config endpoint.
// Never hardcode API keys here — add them to your .env file instead.
async function loadFirebaseConfig() {
  // Allow pre-injection via window.NEUROSCAN_CONFIG (e.g., set by a server-rendered
  // meta tag or a <script> block in index.html for static hosting scenarios)
  if (window.NEUROSCAN_CONFIG && window.NEUROSCAN_CONFIG.apiKey) {
    return window.NEUROSCAN_CONFIG;
  }
  try {
    const res = await fetch('/api/config');
    if (res.ok) {
      const cfg = await res.json();
      if (cfg && cfg.apiKey) return cfg;
    }
  } catch (e) {
    console.warn('Could not load Firebase config from /api/config:', e.message);
  }
  // Final fallback — throw a clear error rather than silently using empty values
  throw new Error(
    'NeuroScan: Firebase configuration not found. ' +
    'Set FIREBASE_API_KEY and related variables in your .env file.'
  );
}

// Bootstrap: load config then initialize Firebase.
// Resolves to { firebaseApp, authInstance } or { firebaseApp: null, authInstance: null }
// when config is unavailable (dev mode without .env or offline).
const _firebaseReady = loadFirebaseConfig().then(FIREBASE_CONFIG => {
  const firebaseApp    = initializeApp(FIREBASE_CONFIG);
  const authInstance   = getAuth(firebaseApp);
  return { firebaseApp, authInstance };
}).catch(err => {
  console.warn(
    '[NeuroScan] Firebase init skipped (running in demo/dev mode without credentials).',
    err.message
  );
  return { firebaseApp: null, authInstance: null };
});

// Backend API URL — default to relative /api on current origin
const API_BASE = window.NEUROSCAN_API_URL || '/api';

// ── FIREBASE AUTH — lazy resolved instance ────────────────────
// _authInstance is populated once _firebaseReady settles. All auth
// functions call _requireAuth() so they fail with a clear message
// instead of "Cannot read properties of null" when Firebase is
// unavailable (e.g. no .env credentials in local dev).
let _authInstance = null;
_firebaseReady.then(r => { _authInstance = r.authInstance; });

function _requireAuth() {
  if (!_authInstance) {
    throw new Error(
      'Firebase Auth is not initialised. ' +
      'Set FIREBASE_API_KEY and the other FIREBASE_* variables in your .env file, ' +
      'then restart the server.'
    );
  }
  return _authInstance;
}

// auth export — kept for pages that reference auth.currentUser directly.
// May be null until _firebaseReady resolves; use _requireAuth() inside
// async functions to get a guaranteed non-null instance.
export const auth = await _firebaseReady.then(r => r.authInstance);
const googleProvider = new GoogleAuthProvider();

// ── API HELPER ────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  // Attach Firebase ID token to every request if available
  const authInst = _authInstance;
  const user  = authInst ? authInst.currentUser : null;
  const token = user ? await user.getIdToken() : null;

  const headers = {
    'Content-Type': 'application/json',
    ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
    ...options.headers,
  };

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API error ${res.status}`);
  }

  return res.json();
}

// ── AUTH HELPERS ──────────────────────────────────────────────
export async function signInWithGoogle() {
  const result = await signInWithPopup(_requireAuth(), googleProvider);
  await ensureUserDoc(result.user);
  return result.user;
}

export async function signInEmail(email, password) {
  const result = await signInWithEmailAndPassword(_requireAuth(), email, password);
  return result.user;
}

export async function signUpEmail(email, password, displayName) {
  const result = await createUserWithEmailAndPassword(_requireAuth(), email, password);
  await updateProfile(result.user, { displayName });
  await ensureUserDoc(result.user, displayName);
  return result.user;
}

export async function firebaseSignOut() {
  await signOut(_requireAuth());
}

export function onAuthChange(callback) {
  // If Firebase hasn't initialised yet (no credentials), wait for it and
  // call the callback with null so pages show the signed-out state correctly
  // instead of hanging indefinitely or throwing.
  if (!_authInstance) {
    _firebaseReady.then(r => {
      if (r.authInstance) {
        onAuthStateChanged(r.authInstance, callback);
      } else {
        callback(null); // no Firebase — behave as signed-out
      }
    });
    return () => {}; // unsubscribe no-op
  }
  return onAuthStateChanged(_authInstance, callback);
}

// ── USER PROFILE ──────────────────────────────────────────────
export async function ensureUserDoc(user, displayName) {
  try {
    await apiFetch(`/users/${user.uid}`, {
      method: 'PUT',
      body: JSON.stringify({
        email:       user.email,
        displayName: displayName || user.displayName || 'User',
        photoURL:    user.photoURL || '',
      }),
    });
  } catch (e) {
    console.warn('ensureUserDoc failed (non-fatal):', e.message);
  }
}

export async function getUserDoc(uid) {
  try {
    return await apiFetch(`/users/${uid}`);
  } catch (e) {
    if (e.message.includes('404')) return null;
    throw e;
  }
}

// ── ASSESSMENTS ───────────────────────────────────────────────
export async function saveAssessment(uid, data) {
  const result = await apiFetch('/assessments', {
    method: 'POST',
    body: JSON.stringify({ ...data, uid }),
  });
  return result.id;
}

export async function getUserAssessments(uid, maxCount = 20) {
  return apiFetch(`/assessments/${uid}?limit=${maxCount}`);
}

export async function deleteAssessment(id) {
  await apiFetch(`/assessments/${id}`, { method: 'DELETE' });
}

export async function getAssessmentById(id) {
  try {
    return await apiFetch(`/assessments/id/${id}`);
  } catch (e) {
    if (e.message.includes('404')) return null;
    throw e;
  }
}

// Real-time listener — MongoDB doesn't natively support browser push.
// We poll every 8 seconds and call the callback when data changes.
export function watchAssessments(uid, callback) {
  let lastCount = -1;

  async function poll() {
    try {
      const data = await getUserAssessments(uid, 10);
      // Only fire callback if something changed
      if (data.length !== lastCount) {
        lastCount = data.length;
        callback(data);
      }
    } catch (e) {
      console.warn('watchAssessments poll error:', e.message);
    }
  }

  // Fire immediately
  poll();
  // Then poll every 8 seconds
  const interval = setInterval(poll, 8000);

  // Return unsubscribe function (matching Firestore API)
  return () => clearInterval(interval);
}

// ── CHAT MESSAGES ─────────────────────────────────────────────
export async function saveChatMessage(uid, role, text) {
  await apiFetch('/chats', {
    method: 'POST',
    body: JSON.stringify({ uid, role, text }),
  });
}

export async function getChatHistory(uid, maxCount = 50) {
  return apiFetch(`/chats/${uid}?limit=${maxCount}`);
}

export async function clearChatHistory(uid) {
  await apiFetch(`/chats/${uid}`, { method: 'DELETE' });
}

// ── REFERRALS ─────────────────────────────────────────────────
export async function createReferral(uid, assessmentId, notes) {
  return apiFetch('/referrals', {
    method: 'POST',
    body: JSON.stringify({ uid, assessmentId, notes }),
  });
}

// ── LEGACY COMPAT ─────────────────────────────────────────────
// serverTimestamp is a Firestore concept — returns a JS Date for MongoDB
export function serverTimestamp() {
  return new Date();
}

// db export kept for any page that imports it (will be null for MongoDB)
export const db = null;
export { googleProvider };
