"""Local dashboard — FastAPI + inline HTML, Claude-style dark UI."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ..config import MemriConfig
from ..core.memory import MemriMemory

_config = MemriConfig.load()
_memory = MemriMemory(_config)

app = FastAPI(title="Memri Dashboard", version="0.1.0")

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Memri</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:       #0a0a0a;
  --surface:  #141414;
  --surface2: #1c1c1c;
  --border:   #242424;
  --border2:  #2e2e2e;
  --text:     #f0ede8;
  --text-2:   #9e9a94;
  --text-3:   #55524e;
  --accent:   #c96a3a;
  --accent-d: rgba(201,106,58,.13);
  --accent-h: #d9784b;
  --green:    #4ade80;
  --green-d:  rgba(74,222,128,.10);
  --r: 10px; --r-sm: 6px;
}

html { height: 100%; }
body {
  font-family: 'Inter', -apple-system, sans-serif;
  background: var(--bg); color: var(--text);
  height: 100%; margin: 0;
  font-size: 14px; line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

/* ── Layout ─────────────────────────────── */
.layout { display: flex; height: 100vh; overflow: hidden; }

/* ── Sidebar ────────────────────────────── */
.sidebar {
  width: 216px; flex-shrink: 0;
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
  overflow: hidden;
}
.s-logo {
  padding: 20px 16px 16px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 10px;
  flex-shrink: 0;
}
.s-logo-icon {
  width: 30px; height: 30px; border-radius: 8px;
  background: var(--accent);
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 15px; color: #fff; flex-shrink: 0;
}
.s-logo-name { font-size: 15px; font-weight: 600; letter-spacing: -.02em; }
.s-logo-ver  { font-size: 10px; color: var(--text-3); margin-top: 1px; }

.s-nav { padding: 10px 8px; flex: 1; overflow-y: auto; }
.s-label {
  font-size: 10px; font-weight: 600; color: var(--text-3);
  text-transform: uppercase; letter-spacing: .08em;
  padding: 10px 10px 4px;
}
.s-btn {
  display: flex; align-items: center; gap: 9px;
  padding: 7px 10px; border-radius: var(--r-sm);
  cursor: pointer; color: var(--text-2);
  font-size: 13.5px; font-weight: 450;
  transition: background .12s, color .12s;
  border: none; background: none; width: 100%; text-align: left; font-family: inherit;
}
.s-btn:hover  { background: var(--surface2); color: var(--text); }
.s-btn.active { background: var(--accent-d); color: var(--accent); font-weight: 500; }
.s-btn svg    { width: 15px; height: 15px; opacity: .75; flex-shrink: 0; }

.s-foot {
  padding: 12px 16px; border-top: 1px solid var(--border);
  font-size: 11px; color: var(--text-3); flex-shrink: 0;
  line-height: 1.5;
}

/* ── Content area ───────────────────────── */
.content-area {
  flex: 1;
  display: flex; flex-direction: column;
  overflow: hidden; /* children scroll, not this */
}

.topbar {
  height: 52px; flex-shrink: 0;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 24px;
  background: rgba(10,10,10,.9);
  backdrop-filter: blur(10px);
  z-index: 5;
}
.topbar-title { font-size: 15px; font-weight: 600; letter-spacing: -.01em; }

.btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 13px; border-radius: var(--r-sm);
  font-size: 13px; font-weight: 500;
  cursor: pointer; border: none; font-family: inherit;
  transition: all .12s;
}
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { background: var(--accent-h); }
.btn-ghost {
  background: transparent; color: var(--text-2);
  border: 1px solid var(--border2);
}
.btn-ghost:hover { background: var(--surface2); color: var(--text); }
.btn-back {
  background: var(--surface2); color: var(--text-2);
  border: 1px solid var(--border2);
  gap: 5px;
}
.btn-back:hover { color: var(--text); background: var(--border2); }

/* ── Pages ───────────────────────────────── */
/* Each page fills and scrolls independently */
.page { display: none; flex: 1; overflow-y: auto; }
.page.active { display: block; }

.page-inner { padding: 24px; max-width: 1000px; }

/* ── Stat cards ──────────────────────────── */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(175px, 1fr));
  gap: 12px; margin-bottom: 22px;
}
.stat-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r); padding: 16px 18px;
}
.stat-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.stat-label { font-size: 11px; font-weight: 500; color: var(--text-3); text-transform: uppercase; letter-spacing: .07em; }
.stat-ico { width: 26px; height: 26px; border-radius: var(--r-sm); display: flex; align-items: center; justify-content: center; font-size: 13px; }
.stat-val { font-size: 24px; font-weight: 700; letter-spacing: -.03em; line-height: 1; margin-bottom: 3px; }
.stat-sub { font-size: 11px; color: var(--text-3); }

/* ── Section box ─────────────────────────── */
.box {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r); margin-bottom: 16px; overflow: hidden;
}
.box-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 18px; border-bottom: 1px solid var(--border);
}
.box-title {
  font-size: 13px; font-weight: 600;
  display: flex; align-items: center; gap: 7px;
}
.box-title svg { width: 14px; height: 14px; opacity: .7; }
.badge {
  font-size: 11px; font-weight: 500;
  background: var(--surface2); color: var(--text-2);
  padding: 2px 7px; border-radius: 20px; border: 1px solid var(--border);
}

/* ── Session rows ────────────────────────── */
.t-row {
  display: flex; align-items: center;
  padding: 11px 18px; gap: 12px;
  border-bottom: 1px solid var(--border);
  cursor: pointer; transition: background .1s; user-select: none;
}
.t-row:last-child { border-bottom: none; }
.t-row:hover { background: rgba(255,255,255,.025); }

.t-dot  { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); flex-shrink: 0; opacity: .5; }
.t-info { flex: 1; min-width: 0; }
.t-id   {
  font-family: 'SF Mono','Cascadia Code','Fira Code',monospace;
  font-size: 12px; color: var(--text-2);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.t-meta { font-size: 11.5px; color: var(--text-3); margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.t-arrow { color: var(--text-3); font-size: 14px; flex-shrink: 0; }

.chip { font-size: 11px; font-weight: 500; padding: 2px 8px; border-radius: 20px; flex-shrink: 0; }
.chip-claude  { background: var(--accent-d); color: var(--accent); }
.chip-cursor  { background: rgba(99,179,237,.12); color: #63b3ed; }
.chip-codex   { background: rgba(154,230,180,.12); color: #68d391; }
.chip-unknown { background: var(--surface2); color: var(--text-3); }

/* ── Search ──────────────────────────────── */
.search-row { display: flex; gap: 8px; padding: 14px 18px; border-bottom: 1px solid var(--border); }
.search-input {
  flex: 1; background: var(--surface2); border: 1px solid var(--border2);
  border-radius: var(--r-sm); padding: 8px 13px;
  color: var(--text); font-size: 13.5px; font-family: inherit;
  outline: none; transition: border-color .15s;
}
.search-input::placeholder { color: var(--text-3); }
.search-input:focus { border-color: var(--accent); }
.search-results {
  padding: 16px 18px; font-size: 13px; color: var(--text-2);
  white-space: pre-wrap; font-family: 'SF Mono','Cascadia Code',monospace;
  min-height: 80px; line-height: 1.7;
}
.search-results mark { background: rgba(201,106,58,.22); color: var(--accent-h); border-radius: 2px; padding: 0 2px; }

/* ── Detail page ─────────────────────────── */
.detail-header {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r); padding: 16px 20px;
  margin-bottom: 16px; display: flex; flex-direction: column; gap: 8px;
}
.detail-meta {
  display: flex; gap: 16px; flex-wrap: wrap;
  font-size: 12px; color: var(--text-3);
}
.detail-meta span { display: flex; align-items: center; gap: 5px; }

/* Tabs for detail page */
.tabs { display: flex; gap: 2px; margin-bottom: 16px; }
.tab {
  padding: 7px 16px; border-radius: var(--r-sm);
  font-size: 13px; font-weight: 500; cursor: pointer;
  border: none; background: var(--surface2); color: var(--text-3);
  font-family: inherit; transition: all .12s;
  border: 1px solid var(--border);
}
.tab:hover { color: var(--text-2); }
.tab.active { background: var(--accent-d); color: var(--accent); border-color: rgba(201,106,58,.3); }

/* ── Message list ────────────────────────── */
.msg-list { display: flex; flex-direction: column; gap: 10px; }

.msg {
  border-radius: var(--r);
  padding: 12px 15px;
  line-height: 1.65;
}
.msg-user {
  background: var(--accent-d);
  border: 1px solid rgba(201,106,58,.2);
  margin-left: 40px;
}
.msg-assistant {
  background: var(--surface);
  border: 1px solid var(--border);
  margin-right: 40px;
}
.msg-tool {
  background: rgba(251,191,36,.06);
  border: 1px solid rgba(251,191,36,.15);
  margin-right: 80px; margin-left: 80px;
}
.msg-system {
  background: var(--surface2); border: 1px solid var(--border);
  font-style: italic; opacity: .7;
}

.msg-role {
  font-size: 10.5px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .07em;
  margin-bottom: 6px;
}
.msg-user .msg-role      { color: var(--accent); }
.msg-assistant .msg-role { color: var(--text-3); }
.msg-tool .msg-role      { color: #fbbf24; }
.msg-system .msg-role    { color: var(--text-3); }

.msg-content {
  color: var(--text-2);
  font-size: 13.5px;
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
}
.msg-time { font-size: 10.5px; color: var(--text-3); margin-top: 6px; text-align: right; }

/* ── Observations ────────────────────────── */
.obs-block {
  padding: 18px 20px;
  font-family: 'SF Mono','Cascadia Code','Fira Code',monospace;
  font-size: 12.5px; line-height: 1.8;
  color: var(--text-2); white-space: pre-wrap; word-break: break-word;
}
.obs-date   { color: var(--accent); font-weight: 600; }
.obs-red    { color: #f87171; }
.obs-yellow { color: #fbbf24; }

/* ── Empty ───────────────────────────────── */
.empty {
  display: flex; flex-direction: column; align-items: center;
  padding: 52px 20px; gap: 10px; text-align: center;
}
.empty-ico { width: 44px; height: 44px; border-radius: 11px; background: var(--surface2); border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; font-size: 20px; }
.empty h3  { font-size: 14px; font-weight: 600; }
.empty p   { font-size: 12.5px; color: var(--text-3); max-width: 300px; line-height: 1.6; }
.empty code { font-family: monospace; background: var(--surface2); border: 1px solid var(--border); border-radius: var(--r-sm); padding: 5px 12px; font-size: 12px; color: var(--accent); display: inline-block; margin-top: 4px; }

/* ── Spinner ─────────────────────────────── */
.spin { width: 18px; height: 18px; border: 2px solid var(--border2); border-top-color: var(--accent); border-radius: 50%; animation: rot .7s linear infinite; display: inline-block; }
@keyframes rot { to { transform: rotate(360deg); } }

/* ── Scrollbars ──────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }
</style>
</head>
<body>
<div class="layout">

<!-- Sidebar -->
<aside class="sidebar">
  <div class="s-logo">
    <div class="s-logo-icon">m</div>
    <div>
      <div class="s-logo-name">memri</div>
      <div class="s-logo-ver">v1.0.0 &middot; graph memory</div>
    </div>
  </div>
  <nav class="s-nav">
    <div class="s-label">Overview</div>
    <button class="s-btn active" onclick="goPage(this,'overview')">
      <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"/></svg>
      Overview
    </button>
    <div class="s-label" style="margin-top:10px">Memory</div>
    <button class="s-btn" data-page="sessions" onclick="goPage(this,'sessions')">
      <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>
      Sessions
    </button>
    <button class="s-btn" onclick="goPage(this,'search')">
      <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
      Search
    </button>
    <button class="s-btn" onclick="goPage(this,'memory-graph')">
      <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/><path stroke-linecap="round" stroke-linejoin="round" d="M12 7v5m0 0l-5 5m5-5l5 5"/></svg>
      Memory Graph
    </button>
    <button class="s-btn" onclick="goPage(this,'layer0')">
      <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>
      Layer 0
    </button>
    <button class="s-btn" onclick="goPage(this,'episodes')">
      <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4"/></svg>
      Episodes
    </button>
    <button class="s-btn" onclick="goPage(this,'timeline')">
      <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M4 6h16M4 10h16M4 14h16M4 18h16"/></svg>
      Timeline
    </button>
    <div class="s-label" style="margin-top:10px">System</div>
    <button class="s-btn" onclick="goPage(this,'settings')">
      <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"><path stroke-linecap="round" stroke-linejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
      Settings
    </button>
  </nav>
  <div class="s-foot" id="s-foot">Loading&hellip;</div>
</aside>

<!-- Content -->
<div class="content-area">

  <header class="topbar">
    <div id="topbar-left" style="display:flex;align-items:center;gap:10px">
      <div class="topbar-title" id="page-title">Overview</div>
    </div>
    <button class="btn btn-ghost" onclick="reload()" style="font-size:12px;padding:5px 11px">
      <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
      Refresh
    </button>
  </header>

  <!-- Overview -->
  <div class="page active" id="page-overview">
    <div class="page-inner">
      <div class="stats-grid" id="stats-grid">
        <div class="stat-card"><div class="stat-val"><div class="spin"></div></div></div>
      </div>
      <div class="box">
        <div class="box-head">
          <span class="box-title">
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            Recent Sessions
          </span>
          <button class="btn btn-ghost" style="font-size:12px;padding:4px 10px"
            onclick="goPage(document.querySelector('.s-btn:nth-child(2)'),'sessions')">
            View all &rarr;
          </button>
        </div>
        <div id="overview-list"><div class="empty"><div class="spin"></div></div></div>
      </div>
    </div>
  </div>

  <!-- Sessions list -->
  <div class="page" id="page-sessions">
    <div class="page-inner">
      <div class="box">
        <div class="box-head">
          <span class="box-title">
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>
            All Sessions
            <span class="badge" id="sessions-count">–</span>
          </span>
        </div>
        <div id="sessions-list"><div class="empty"><div class="spin"></div></div></div>
      </div>
    </div>
  </div>

  <!-- Session detail -->
  <div class="page" id="page-detail">
    <div class="page-inner">
      <div class="detail-header" id="detail-header"></div>
      <div class="tabs">
        <button class="tab active" id="tab-msgs" onclick="switchTab('msgs')">Messages</button>
        <button class="tab"        id="tab-obs"  onclick="switchTab('obs')">Observations</button>
      </div>
      <div id="detail-body"></div>
    </div>
  </div>

  <!-- Timeline -->
  <div class="page" id="page-timeline">
    <div class="page-inner">
      <div class="box" id="timeline-box">
        <div class="box-head">
          <span class="box-title">
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M4 6h16M4 10h16M4 14h16M4 18h16"/></svg>
            Observation Timeline
            <span class="badge" id="timeline-count">–</span>
          </span>
        </div>
        <div id="timeline-list"><div class="empty"><div class="spin"></div></div></div>
      </div>
    </div>
  </div>

  <!-- Settings -->
  <div class="page" id="page-settings">
    <div class="page-inner">
      <div class="box" style="margin-bottom:16px">
        <div class="box-head"><span class="box-title">
          <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
          Configuration
        </span></div>
        <div id="settings-form" style="padding:20px">
          <div class="spin"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Search -->
  <div class="page" id="page-search">
    <div class="page-inner">
      <div class="box">
        <div class="box-head">
          <span class="box-title">
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
            Search Memories
          </span>
        </div>
        <div class="search-row">
          <input class="search-input" id="search-q" type="text"
            placeholder="Search across all past sessions&hellip;"
            onkeydown="if(event.key==='Enter')doSearch()">
          <button class="btn btn-primary" onclick="doSearch()">Search</button>
        </div>
        <div class="search-results" id="search-results" style="color:var(--text-3);font-family:inherit">
          Type a query and press Enter.
        </div>
      </div>
    </div>
  </div>

  <!-- Memory Graph page -->
  <div class="page" id="page-memory-graph">
    <div class="page-inner">
      <div class="box" style="margin-bottom:16px">
        <div class="box-head">
          <span class="box-title">Memory Graph</span>
          <span class="badge" id="graph-node-count">–</span>
        </div>
        <div style="padding:12px 18px;font-size:12px;color:var(--text-3)">
          Click a node to inspect its content. Colors: <span style="color:#60a5fa">fact</span> &nbsp; <span style="color:#fb923c">entity</span> &nbsp; <span style="color:#a78bfa">reflection</span> &nbsp; <span style="color:#6b7280">episode</span>
        </div>
        <canvas id="graph-canvas" style="width:100%;height:520px;display:block;background:var(--surface2);border-top:1px solid var(--border)"></canvas>
      </div>
      <div class="box" id="graph-node-detail" style="display:none">
        <div class="box-head"><span class="box-title">Node Detail</span></div>
        <div id="graph-node-content" style="padding:16px 18px;font-size:13px;color:var(--text-2);line-height:1.6"></div>
      </div>
    </div>
  </div>

  <!-- Layer 0 page -->
  <div class="page" id="page-layer0">
    <div class="page-inner">
      <div class="box" style="margin-bottom:16px">
        <div class="box-head"><span class="box-title">Memory Index (Layer 0)</span></div>
        <div id="layer0-content" style="padding:16px 18px">
          <div class="spin"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Episodes page -->
  <div class="page" id="page-episodes">
    <div class="page-inner">
      <div class="box">
        <div class="box-head">
          <span class="box-title">Episodes (Layer 2)</span>
          <span class="badge" id="episodes-count">–</span>
        </div>
        <div id="episodes-list" style="padding:0">
          <div class="empty"><div class="spin"></div></div>
        </div>
      </div>
    </div>
  </div>

</div><!-- /content-area -->
</div><!-- /layout -->

<script>
// ── State ─────────────────────────────────────────────────────────
let _threads = [];
let _currentThread = null;
let _currentTab = 'msgs';

// ── Navigation ────────────────────────────────────────────────────
const TITLES = { overview:'Overview', sessions:'Sessions', detail:'Session Detail', search:'Search Memories', timeline:'Observation Timeline', settings:'Settings', 'memory-graph':'Memory Graph', layer0:'Memory Index', episodes:'Episodes (Layer 2)' };

function goPage(btn, id) {
  document.querySelectorAll('.s-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => {
    p.classList.remove('active');
    p.scrollTop = 0;
  });

  if (btn) btn.classList.add('active');
  document.getElementById('page-' + id).classList.add('active');
  document.getElementById('page-title').textContent = TITLES[id] || id;

  // Show/hide back button
  const backBtn = document.getElementById('back-btn');
  if (id === 'detail') {
    if (!backBtn) {
      const b = document.createElement('button');
      b.id = 'back-btn';
      b.className = 'btn btn-back';
      b.innerHTML = '&#8592; Back';
      b.onclick = () => goPage(document.querySelector('[data-page="sessions"]'), 'sessions');
      document.getElementById('topbar-left').prepend(b);
    }
  } else {
    if (backBtn) backBtn.remove();
  }

  if (id === 'sessions') loadSessions();
  if (id === 'timeline') loadTimeline();
  if (id === 'settings') loadSettings();
  if (id === 'memory-graph') loadMemoryGraph();
  if (id === 'layer0') loadLayer0();
  if (id === 'episodes') loadEpisodes();
}

// ── API ───────────────────────────────────────────────────────────
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

// ── Stats ─────────────────────────────────────────────────────────
async function loadStats() {
  const d = await api('/api/stats');
  const fmt = n => n >= 1e6 ? (n/1e6).toFixed(1)+'M' : n >= 1e3 ? (n/1e3).toFixed(1)+'K' : String(n);
  const net = d.cost_saved_usd - d.llm_cost_usd;

  document.getElementById('s-foot').innerHTML =
    d.threads + ' sessions<br>$' + d.cost_saved_usd.toFixed(4) + ' saved';

  document.getElementById('stats-grid').innerHTML = `
    <div class="stat-card">
      <div class="stat-top"><span class="stat-label">Sessions</span>
        <div class="stat-ico" style="background:var(--accent-d);color:var(--accent)">&#128172;</div></div>
      <div class="stat-val">${d.threads}</div>
      <div class="stat-sub">${d.messages.toLocaleString()} messages</div>
    </div>
    <div class="stat-card">
      <div class="stat-top"><span class="stat-label">Tokens Saved</span>
        <div class="stat-ico" style="background:var(--green-d);color:var(--green)">&#9889;</div></div>
      <div class="stat-val">${fmt(d.tokens_saved)}</div>
      <div class="stat-sub">${d.observations} observation blocks</div>
    </div>
    <div class="stat-card">
      <div class="stat-top"><span class="stat-label">Cost Saved</span>
        <div class="stat-ico" style="background:var(--green-d);color:var(--green)">&#128176;</div></div>
      <div class="stat-val">$${d.cost_saved_usd.toFixed(4)}</div>
      <div class="stat-sub">vs full context</div>
    </div>
    <div class="stat-card">
      <div class="stat-top"><span class="stat-label">Memri Cost</span>
        <div class="stat-ico" style="background:var(--surface2);color:var(--text-3)">&#129302;</div></div>
      <div class="stat-val">$${d.llm_cost_usd.toFixed(4)}</div>
      <div class="stat-sub">observer + reflector</div>
    </div>
    <div class="stat-card">
      <div class="stat-top"><span class="stat-label">Net Savings</span>
        <div class="stat-ico" style="background:var(--accent-d);color:var(--accent)">&#128200;</div></div>
      <div class="stat-val" style="color:${net>=0?'var(--green)':'var(--red)'}">${net>=0?'+':''}$${Math.abs(net).toFixed(4)}</div>
      <div class="stat-sub">${net>=0?'Positive ROI':'Builds up over time'}</div>
    </div>
  `;
}

// ── Sessions ──────────────────────────────────────────────────────
async function loadSessions() {
  _threads = await api('/api/threads');
  document.getElementById('sessions-count').textContent = _threads.length;
  document.getElementById('sessions-list').innerHTML  = renderRows(_threads, 999);
  document.getElementById('overview-list').innerHTML  = renderRows(_threads, 5);
}

function renderRows(list, limit) {
  if (!list.length) return `
    <div class="empty">
      <div class="empty-ico">&#128173;</div>
      <h3>No sessions yet</h3>
      <p>Ingest your existing Claude Code sessions to get started.</p>
      <code>memri ingest --agent claude-code</code>
    </div>`;

  const rows = list.slice(0, limit).map(t => {
    const agent = t.agent_type || 'unknown';
    const chipCls = agent.includes('claude') ? 'chip-claude'
                  : agent === 'cursor'        ? 'chip-cursor'
                  : agent === 'codex'         ? 'chip-codex'
                  : 'chip-unknown';
    const label = agent.replace('claude-code','Claude Code');
    const project = t.project_path
      ? t.project_path.replace(/\\\\/g,'/').split('/').filter(Boolean).slice(-2).join('/')
      : '—';
    const date = t.updated_at
      ? new Date(t.updated_at).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'})
      : '—';
    return `
      <div class="t-row" onclick="openDetail('${t.id.replace(/'/g,'')}')">
        <div class="t-dot"></div>
        <div class="t-info">
          <div class="t-id">${t.id}</div>
          <div class="t-meta">${project}&nbsp;&middot;&nbsp;${date}</div>
        </div>
        <span class="chip ${chipCls}">${label}</span>
        <span class="t-arrow">&#8250;</span>
      </div>`;
  }).join('');

  const more = list.length > limit
    ? `<div style="padding:10px 18px;font-size:12px;color:var(--text-3)">+ ${list.length - limit} more sessions</div>`
    : '';
  return rows + more;
}

// ── Session detail (full page) ────────────────────────────────────
async function openDetail(threadId) {
  _currentThread = threadId;
  _currentTab = 'msgs';

  const t = _threads.find(x => x.id === threadId) || {};
  const agent = (t.agent_type || 'unknown').replace('claude-code','Claude Code');
  const project = t.project_path
    ? t.project_path.replace(/\\\\/g,'/').split('/').filter(Boolean).slice(-2).join('/')
    : '—';
  const date = t.updated_at ? new Date(t.updated_at).toLocaleDateString() : '—';

  document.getElementById('detail-header').innerHTML = `
    <div style="font-size:13.5px;font-weight:600;color:var(--text)">${threadId}</div>
    <div class="detail-meta">
      <span>&#128172; ${agent}</span>
      <span>&#128194; ${t.project_path || '—'}</span>
      <span>&#128197; ${date}</span>
    </div>`;

  document.getElementById('tab-msgs').classList.add('active');
  document.getElementById('tab-obs').classList.remove('active');

  goPage(null, 'detail');
  await loadDetailTab('msgs');
}

async function switchTab(tab) {
  _currentTab = tab;
  document.getElementById('tab-msgs').classList.toggle('active', tab === 'msgs');
  document.getElementById('tab-obs').classList.toggle('active',  tab === 'obs');
  document.getElementById('detail-body').innerHTML = '<div class="empty"><div class="spin"></div></div>';
  await loadDetailTab(tab);
}

async function loadDetailTab(tab) {
  const body = document.getElementById('detail-body');

  if (tab === 'msgs') {
    try {
      const d = await api('/api/messages/' + _currentThread);
      if (!d.messages.length) {
        body.innerHTML = '<div class="empty"><div class="empty-ico">&#128196;</div><h3>No messages</h3></div>';
        return;
      }
      const items = d.messages.map(m => {
        const safe = (m.content || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        const role = m.role || 'unknown';
        const cls  = role === 'user' ? 'msg-user'
                   : role === 'assistant' ? 'msg-assistant'
                   : role === 'tool' ? 'msg-tool'
                   : 'msg-system';
        const label = role.charAt(0).toUpperCase() + role.slice(1);
        const time  = m.created_at
          ? new Date(m.created_at).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})
          : '';
        return `<div class="msg ${cls}">
          <div class="msg-role">${label}</div>
          <div class="msg-content">${safe}</div>
          ${time ? '<div class="msg-time">'+time+'</div>' : ''}
        </div>`;
      }).join('');
      body.innerHTML = `<div class="msg-list">${items}</div>`;
    } catch(e) {
      body.innerHTML = '<div class="empty"><div class="empty-ico">&#128196;</div><h3>No messages found</h3></div>';
    }

  } else {
    try {
      const d = await api('/api/observations/' + _currentThread);
      body.innerHTML =
        '<div class="box"><div class="obs-block">' + colorizeObs(d.content) + '</div>'
        + '<div style="padding:10px 18px;font-size:11px;color:var(--text-3);border-top:1px solid var(--border)">'
        + d.token_count.toLocaleString() + ' tokens &nbsp;&middot;&nbsp; version ' + d.version
        + '</div></div>';
    } catch(e) {
      body.innerHTML = `
        <div class="box"><div class="empty">
          <div class="empty-ico">&#129504;</div>
          <h3>No observations yet</h3>
          <p>Observations are generated once a thread exceeds 30K tokens. They compress your conversation history into dense, searchable memory.</p>
        </div></div>`;
    }
  }
}

function colorizeObs(text) {
  return text.split('\\n').map(line => {
    const e = line.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    if (/^Date:/.test(line))  return '<span class="obs-date">' + e + '</span>';
    if (line.includes('\\ud83d\\udd34') || e.includes('&#128308;') || /🔴/.test(line))
      return '<span class="obs-red">' + e + '</span>';
    if (/🟡/.test(line)) return '<span class="obs-yellow">' + e + '</span>';
    return e;
  }).join('\\n');
}

// ── Search ────────────────────────────────────────────────────────
async function doSearch() {
  const q = document.getElementById('search-q').value.trim();
  if (!q) return;
  const el = document.getElementById('search-results');
  el.innerHTML = '<div class="spin"></div>';
  el.style.fontFamily = 'inherit';
  const d = await api('/api/search?q=' + encodeURIComponent(q));
  if (!d.results || d.results.includes('No relevant')) {
    el.style.color = 'var(--text-3)';
    el.innerHTML = 'No relevant memories found for &ldquo;' + q + '&rdquo;.';
  } else {
    el.style.color = 'var(--text-2)';
    el.style.fontFamily = "'SF Mono','Cascadia Code',monospace";
    const esc = d.results.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    el.innerHTML = esc.replace(
      new RegExp(q.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&'), 'gi'),
      m => '<mark>' + m + '</mark>'
    );
  }
}

// ── Timeline ─────────────────────────────────────────────────────
async function loadTimeline() {
  const list = document.getElementById('timeline-list');
  list.innerHTML = '<div class="empty"><div class="spin"></div></div>';
  const data = await api('/api/timeline');
  document.getElementById('timeline-count').textContent = data.length;

  if (!data.length) {
    list.innerHTML = `<div class="empty">
      <div class="empty-ico">&#129504;</div>
      <h3>No observations yet</h3>
      <p>Run <code>memri observe</code> to compress your session history into observation blocks.</p>
      <code>memri observe</code>
    </div>`;
    return;
  }

  list.innerHTML = data.map(o => {
    const project = o.project_path
      ? o.project_path.replace(/\\\\/g,'/').split('/').filter(Boolean).slice(-2).join('/')
      : '—';
    const agent = (o.agent_type || 'unknown').replace('claude-code','Claude Code');
    const date = o.created_at ? new Date(o.created_at).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'}) : '—';
    const preview = o.content.split('\\n').slice(0,4).join('\\n');
    const previewEsc = preview.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const colored = previewEsc.split('\\n').map(line => {
      if (/^Date:/.test(line)) return '<span style="color:var(--accent);font-weight:600">'+line+'</span>';
      if (/🔴/.test(line))     return '<span style="color:#f87171">'+line+'</span>';
      if (/🟡/.test(line))     return '<span style="color:#fbbf24">'+line+'</span>';
      return line;
    }).join('\\n');

    const tid = o.thread_id.replace(/'/g,'');
    return `
      <div style="border-bottom:1px solid var(--border);padding:16px 18px" class="tl-item">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
          <div style="display:flex;align-items:center;gap:10px">
            <div style="width:8px;height:8px;border-radius:50%;background:var(--accent);flex-shrink:0"></div>
            <span style="font-family:monospace;font-size:12px;color:var(--text-2)">${o.thread_id.slice(0,12)}…</span>
            <span style="font-size:11px;color:var(--text-3)">${project}</span>
          </div>
          <div style="display:flex;align-items:center;gap:10px">
            <span style="font-size:11px;color:var(--text-3)">${o.token_count.toLocaleString()} tokens</span>
            <span style="font-size:11px;color:var(--text-3)">${date}</span>
            <button class="btn btn-ghost" style="font-size:11px;padding:3px 9px" onclick="openDetail('${tid}')">Open</button>
          </div>
        </div>
        <pre style="font-family:'SF Mono','Cascadia Code',monospace;font-size:11.5px;line-height:1.6;color:var(--text-3);white-space:pre-wrap;word-break:break-word">${colored}</pre>
      </div>`;
  }).join('');
}

// ── Settings ──────────────────────────────────────────────────────
async function loadSettings() {
  const form = document.getElementById('settings-form');
  form.innerHTML = '<div class="spin"></div>';
  const cfg = await api('/api/config');

  const fields = [
    { key:'observe_threshold', label:'Observe threshold (tokens)', type:'number', hint:'Messages are compressed once unobserved tokens exceed this.' },
    { key:'reflect_threshold', label:'Reflect threshold (tokens)', type:'number', hint:'Observations are garbage-collected when they exceed this.' },
    { key:'llm_provider',      label:'LLM provider', type:'select', opts:['anthropic','gemini','openai','openai-compatible'], hint:'Provider for Observer and Reflector agents. Use "openai-compatible" for Groq, Mistral, Ollama, Together, etc.' },
    { key:'llm_model',         label:'LLM model',    type:'text',   hint:'Model ID — e.g. gemini-2.5-flash, claude-haiku-4-5-20251001, gpt-4o-mini, llama-3.3-70b-versatile' },
    { key:'llm_base_url',      label:'LLM base URL', type:'text',   hint:'For openai-compatible only. Groq: https://api.groq.com/openai/v1  |  Ollama: http://localhost:11434/v1  |  Together: https://api.together.xyz/v1' },
    { key:'dashboard_port',    label:'Dashboard port', type:'number', hint:'Port for the local dashboard server.' },
  ];

  form.innerHTML = `
    <div style="display:grid;gap:18px;max-width:600px">
      ${fields.map(f => {
        const val = cfg[f.key] ?? '';
        const input = f.type === 'select'
          ? `<select id="cfg-${f.key}" style="background:var(--surface2);border:1px solid var(--border2);border-radius:var(--r-sm);padding:7px 11px;color:var(--text);font-size:13px;font-family:inherit;outline:none;width:100%">
              ${f.opts.map(o => `<option value="${o}"${o===val?' selected':''}>${o}</option>`).join('')}
             </select>`
          : `<input id="cfg-${f.key}" type="${f.type}" value="${val}"
               style="background:var(--surface2);border:1px solid var(--border2);border-radius:var(--r-sm);padding:7px 11px;color:var(--text);font-size:13px;font-family:inherit;outline:none;width:100%;transition:border-color .15s"
               onfocus="this.style.borderColor='var(--accent)'" onblur="this.style.borderColor='var(--border2)'">`;
        return `<div>
          <div style="font-size:12.5px;font-weight:500;color:var(--text);margin-bottom:6px">${f.label}</div>
          ${input}
          <div style="font-size:11.5px;color:var(--text-3);margin-top:4px">${f.hint}</div>
        </div>`;
      }).join('')}
      <div style="display:flex;gap:8px;margin-top:4px">
        <button class="btn btn-primary" onclick="saveSettings()">Save Settings</button>
        <span id="settings-msg" style="font-size:12.5px;color:var(--green);align-self:center;display:none">Saved.</span>
      </div>
    </div>`;
}

async function saveSettings() {
  const keys = ['observe_threshold','reflect_threshold','llm_provider','llm_model','llm_base_url','dashboard_port'];
  const body = {};
  for (const k of keys) {
    const el = document.getElementById('cfg-' + k);
    if (el) body[k] = el.value;
  }
  await fetch('/api/config', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  const msg = document.getElementById('settings-msg');
  msg.style.display = 'inline';
  setTimeout(() => msg.style.display = 'none', 2500);
}

// ── Memory Graph ─────────────────────────────────────────────────
let _graphData = null;
async function loadMemoryGraph() {
  try {
    _graphData = await api('/api/graph');
  } catch(e) {
    document.getElementById('graph-canvas').style.display = 'none';
    document.getElementById('graph-node-count').textContent = 'unavailable';
    return;
  }
  const nodes = _graphData.nodes || [];
  const edges = _graphData.edges || [];
  document.getElementById('graph-node-count').textContent = nodes.length + ' nodes';
  const canvas = document.getElementById('graph-canvas');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width || 800;
  canvas.height = 520;
  const ctx = canvas.getContext('2d');

  // Assign positions with simple force-directed layout
  const W = canvas.width, H = canvas.height;
  const pos = {};
  nodes.forEach((n, i) => {
    const angle = (i / nodes.length) * 2 * Math.PI;
    const r = Math.min(W, H) * 0.35;
    pos[n.id] = { x: W/2 + r * Math.cos(angle), y: H/2 + r * Math.sin(angle), vx: 0, vy: 0 };
  });

  // Simple force simulation
  const nodeMap = {};
  nodes.forEach(n => nodeMap[n.id] = n);
  for (let iter = 0; iter < 120; iter++) {
    // Repulsion
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i+1; j < nodes.length; j++) {
        const a = pos[nodes[i].id], b = pos[nodes[j].id];
        const dx = b.x - a.x, dy = b.y - a.y;
        const d = Math.sqrt(dx*dx + dy*dy) || 1;
        const f = 800 / (d*d);
        a.vx -= f*dx/d; a.vy -= f*dy/d;
        b.vx += f*dx/d; b.vy += f*dy/d;
      }
    }
    // Attraction along edges
    edges.forEach(e => {
      const a = pos[e.source], b = pos[e.target];
      if (!a || !b) return;
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.sqrt(dx*dx + dy*dy) || 1;
      const f = d * 0.01;
      a.vx += f*dx/d; a.vy += f*dy/d;
      b.vx -= f*dx/d; b.vy -= f*dy/d;
    });
    // Center gravity + damping
    nodes.forEach(n => {
      const p = pos[n.id];
      p.vx += (W/2 - p.x) * 0.002;
      p.vy += (H/2 - p.y) * 0.002;
      p.x += p.vx * 0.5; p.y += p.vy * 0.5;
      p.vx *= 0.8; p.vy *= 0.8;
      p.x = Math.max(20, Math.min(W-20, p.x));
      p.y = Math.max(20, Math.min(H-20, p.y));
    });
  }

  const COLOR = { fact:'#60a5fa', entity:'#fb923c', reflection:'#a78bfa', episode:'#6b7280' };

  function draw() {
    ctx.clearRect(0, 0, W, H);
    // Draw edges
    ctx.strokeStyle = 'rgba(100,100,100,0.3)';
    ctx.lineWidth = 1;
    edges.forEach(e => {
      const a = pos[e.source], b = pos[e.target];
      if (!a || !b) return;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    });
    // Draw nodes
    nodes.forEach(n => {
      const p = pos[n.id];
      const r = n.type === 'entity' ? 7 : n.type === 'episode' ? 5 : 6;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, 2*Math.PI);
      ctx.fillStyle = COLOR[n.type] || '#888';
      ctx.fill();
      if (n.importance > 0.7) {
        ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1.5; ctx.stroke();
      }
    });
  }
  draw();

  // Click handler
  canvas.onclick = (ev) => {
    const rect2 = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect2.left, my = ev.clientY - rect2.top;
    for (const n of nodes) {
      const p = pos[n.id];
      if (!p) continue;
      const dx = p.x - mx, dy = p.y - my;
      if (dx*dx + dy*dy < 100) {
        const detail = document.getElementById('graph-node-detail');
        detail.style.display = 'block';
        document.getElementById('graph-node-content').innerHTML =
          '<b>' + n.type.toUpperCase() + '</b> &nbsp; <span style="color:var(--text-3);font-size:11px">' + n.id + '</span><br><br>' +
          '<span style="white-space:pre-wrap">' + (n.content||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</span>' +
          '<br><br><span style="color:var(--text-3);font-size:11px">importance: ' + (n.importance||0).toFixed(2) +
          ' &nbsp; session: ' + (n.session_index ?? '—') + '</span>';
        break;
      }
    }
  };
}

// ── Layer 0 ───────────────────────────────────────────────────────
async function loadLayer0() {
  const el = document.getElementById('layer0-content');
  el.innerHTML = '<div class="spin"></div>';
  let d;
  try { d = await api('/api/layer0'); } catch(e) {
    el.innerHTML = '<div class="empty"><div class="empty-ico">&#128203;</div><h3>Graph engine not active</h3><p>Enable graph memory or ingest sessions to see the memory index.</p></div>';
    return;
  }
  const esc = s => String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  let html = '<div style="display:grid;gap:14px;max-width:700px">';
  if (d.user_summary) html += '<div class="box" style="padding:14px 18px"><b>User Summary</b><p style="color:var(--text-2);margin-top:6px;line-height:1.6">' + esc(d.user_summary) + '</p></div>';
  if (d.emotional_state) html += '<div class="box" style="padding:14px 18px"><b>Emotional State</b><p style="color:var(--text-2);margin-top:6px">' + esc(d.emotional_state) + '</p></div>';
  if (d.active_topics && d.active_topics.length) html += '<div class="box" style="padding:14px 18px"><b>Active Topics</b><p style="color:var(--text-2);margin-top:6px">' + d.active_topics.map(t => '<span class="badge" style="margin-right:4px">'+esc(t)+'</span>').join('') + '</p></div>';
  if (d.entity_index && Object.keys(d.entity_index).length) {
    html += '<div class="box" style="padding:14px 18px"><b>Entity Index</b><div style="margin-top:10px;display:grid;gap:6px">';
    for (const [name, ids] of Object.entries(d.entity_index).slice(0,30)) {
      html += '<div style="display:flex;align-items:center;gap:10px"><span style="color:var(--accent);font-weight:500;min-width:120px">' + esc(name) + '</span><span style="color:var(--text-3);font-size:12px">' + ids.length + ' fact(s)</span></div>';
    }
    html += '</div></div>';
  }
  html += '<div style="font-size:11px;color:var(--text-3)">Facts: ' + (d.fact_count||0) + ' &nbsp; Reflections: ' + (d.reflection_count||0) + ' &nbsp; Last updated: ' + (d.last_updated||'').slice(0,19) + '</div>';
  html += '</div>';
  el.innerHTML = html;
}

// ── Episodes ─────────────────────────────────────────────────────
async function loadEpisodes() {
  const el = document.getElementById('episodes-list');
  el.innerHTML = '<div class="empty"><div class="spin"></div></div>';
  let data;
  try { data = await api('/api/episodes'); } catch(e) {
    el.innerHTML = '<div class="empty"><div class="empty-ico">&#128196;</div><h3>Graph engine not active</h3></div>';
    return;
  }
  document.getElementById('episodes-count').textContent = data.length;
  if (!data.length) {
    el.innerHTML = '<div class="empty"><div class="empty-ico">&#128196;</div><h3>No episodes yet</h3><p>Episodes are stored when the graph engine ingests conversations.</p></div>';
    return;
  }
  el.innerHTML = data.map(ep => {
    const preview = (ep.raw_text||'').slice(0, 120).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const full = (ep.raw_text||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    return '<div style="border-bottom:1px solid var(--border);padding:12px 18px">' +
      '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">' +
      '<span style="font-family:monospace;font-size:12px;color:var(--text-2)">#' + (ep.session_index??0) + ' &nbsp; ' + (ep.session_date||'—') + '</span>' +
      '<span style="font-size:11px;color:var(--text-3)">' + (ep.id||'').slice(0,8) + '</span></div>' +
      '<details><summary style="font-size:12.5px;color:var(--text-3);cursor:pointer">' + preview + '…</summary>' +
      '<pre style="font-size:12px;line-height:1.6;color:var(--text-2);white-space:pre-wrap;word-break:break-word;margin-top:8px">' + full + '</pre></details>' +
      '</div>';
  }).join('');
}

// ── Reload ────────────────────────────────────────────────────────
function reload() {
  loadStats();
  loadSessions();
  const active = document.querySelector('.page.active');
  if (active && active.id === 'page-timeline') loadTimeline();
  if (active && active.id === 'page-settings') loadSettings();
  if (active && active.id === 'page-memory-graph') loadMemoryGraph();
  if (active && active.id === 'page-layer0') loadLayer0();
  if (active && active.id === 'page-episodes') loadEpisodes();
}

// ── Boot ──────────────────────────────────────────────────────────
loadStats();
loadSessions();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
@app.get("/sessions", response_class=HTMLResponse)
@app.get("/search", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return HTMLResponse(_HTML)


@app.get("/api/stats")
async def api_stats():
    return JSONResponse(_memory.get_stats())


@app.get("/api/threads")
async def api_threads():
    threads = _memory.store.list_threads()
    return JSONResponse([
        {
            "id": t.id,
            "agent_type": t.agent_type,
            "project_path": t.project_path,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "updated_at": t.updated_at.isoformat() if t.updated_at else None,
        }
        for t in threads
    ])


@app.get("/api/observations/{thread_id}")
async def api_observation(thread_id: str):
    obs = _memory.store.get_observation(thread_id)
    if not obs:
        raise HTTPException(404, "No observations for this thread")
    return JSONResponse({
        "thread_id": thread_id,
        "content": obs.content,
        "token_count": obs.token_count,
        "version": obs.version,
    })


@app.get("/api/messages/{thread_id}")
async def api_messages(thread_id: str):
    msgs = _memory.store.get_messages(thread_id)
    if not msgs:
        raise HTTPException(404, "No messages for this thread")
    return JSONResponse({
        "thread_id": thread_id,
        "total": len(msgs),
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "token_count": m.token_count,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in msgs
        ],
    })


@app.get("/api/search")
async def api_search(q: str, top_k: int = 5):
    results = _memory.search(q, top_k=top_k)
    return JSONResponse({"query": q, "results": results})


@app.get("/api/timeline")
async def api_timeline():
    """All observations across all threads, sorted newest-first."""
    all_obs = _memory.store.get_all_observations()
    threads = {t.id: t for t in _memory.store.list_threads()}
    return JSONResponse([
        {
            "thread_id": o.thread_id,
            "agent_type": threads.get(o.thread_id, {}).agent_type if o.thread_id in threads else "unknown",
            "project_path": threads.get(o.thread_id, {}).project_path if o.thread_id in threads else "",
            "content": o.content,
            "token_count": o.token_count,
            "version": o.version,
            "created_at": o.created_at.isoformat() if o.created_at else None,
            "reflected_at": o.reflected_at.isoformat() if o.reflected_at else None,
        }
        for o in sorted(all_obs, key=lambda x: x.created_at or __import__("datetime").datetime.min, reverse=True)
    ])


@app.get("/api/config")
async def api_config_get():
    import dataclasses
    data = dataclasses.asdict(_config)
    data.pop("llm_api_key", None)
    return JSONResponse(data)


@app.post("/api/config")
async def api_config_post(request: Request):
    import dataclasses
    body = await request.json()
    allowed = {f.name for f in dataclasses.fields(_config)} - {"llm_api_key"}
    for key, value in body.items():
        if key in allowed and hasattr(_config, key):
            current = getattr(_config, key)
            if isinstance(current, int):
                value = int(value)
            elif isinstance(current, bool):
                value = str(value).lower() in ("1", "true", "yes")
            setattr(_config, key, value)
    _config.save()
    return JSONResponse({"ok": True})


@app.get("/api/graph")
async def api_graph():
    """Return serializable graph data for visualization."""
    engine = _memory.graph_engine
    if not engine:
        raise HTTPException(503, "Graph engine not available")
    return JSONResponse(engine.get_graph_data())


@app.get("/api/layer0")
async def api_layer0():
    """Return Layer 0 (memory index) data."""
    engine = _memory.graph_engine
    if not engine:
        raise HTTPException(503, "Graph engine not available")
    return JSONResponse(engine.get_layer0().model_dump(mode="json"))


@app.get("/api/episodes")
async def api_episodes():
    """Return all Layer 2 episodes."""
    engine = _memory.graph_engine
    if not engine:
        raise HTTPException(503, "Graph engine not available")
    return JSONResponse(engine.get_episodes())


def run(host: str = "127.0.0.1", port: int = 8050) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)
