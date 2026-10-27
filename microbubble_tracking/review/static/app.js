'use strict';
/* Track review UI. All data comes from review_tool.py on this computer; every change is saved by the server. */
const $ = (s) => document.querySelector(s);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL = {};
['--trusted', '--untrusted', '--unreviewed', '--accent', '--wall', '--merge', '--split', '--flag', '--ink', '--ink-2', '--line', '--panel-2']
  .forEach((k) => { COL[k.slice(2)] = css(k); });
const LABEL = { trusted: 'Trusted', untrusted: 'Untrusted', unreviewed: 'Unreviewed' };
const EVENT = { merge: 'Merge', split: 'Split', appear: 'Appears', lost: 'Lost', enter: 'Enters view', exit: 'Leaves view' };

const S = {
  meta: null, N: 0, fps: 60, tracks: [], byId: new Map(), suggestions: [],
  sel: null, detail: null, compare: null, frame: 0, playing: false, speed: 0.5,
  follow: false, zoom: 3, inF: null, outF: null, tlMode: 'track', tab: 'tracks', view: 'object', expanded: new Set(),
  img: new Map(), obj: new Map(), pending: new Map(), hit: [], lastObjs: [], lastImg: null,
};

/* ---------------- network ---------------- */
async function api(path, body) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function lru(map, key, val, max) { map.set(key, val); if (map.size > max) map.delete(map.keys().next().value); }
function cached(kind, f, load, max) {
  const map = kind === 'i' ? S.img : S.obj;
  if (map.has(f)) return Promise.resolve(map.get(f));
  const key = kind + f;
  if (S.pending.has(key)) return S.pending.get(key);
  const p = load().then((v) => { lru(map, f, v, max); S.pending.delete(key); return v; })
    .catch((e) => { S.pending.delete(key); throw e; });
  S.pending.set(key, p);
  return p;
}
const getImage = (f) => cached('i', f, () => fetch('/api/frame/' + f).then((r) => r.blob()).then((b) => createImageBitmap(b)), 400);
const getObjects = (f) => cached('o', f, () => api('/api/objects/' + f), 600);
function prefetch(from, n) {
  for (let f = from; f < Math.min(S.N, from + n); f++) {
    if (S.pending.size > 12) break;
    if (!S.img.has(f)) getImage(f).catch(() => {});
    if (!S.obj.has(f)) getObjects(f).catch(() => {});
  }
}

/* ---------------- messages ---------------- */
let hintTimer = null, sticky = '';
function hint(msg, ms = 2500) {
  const h = $('#hint');
  h.textContent = msg; h.hidden = false;
  clearTimeout(hintTimer);
  hintTimer = setTimeout(() => { hintTimer = null; h.textContent = sticky; h.hidden = !sticky; }, ms);
}
function hintSticky(msg) {
  if (msg === sticky) return;
  sticky = msg;
  if (!hintTimer) { $('#hint').textContent = msg; $('#hint').hidden = !msg; }
}
const tstr = (f) => (f / S.fps).toFixed(3);
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;

/* ---------------- progress & state from the server ---------------- */
function setProgress(p) {
  const t = p.trusted / p.total_rows, u = p.untrusted / p.total_rows;
  $('#pbar .t').style.width = (100 * t).toFixed(2) + '%';
  $('#pbar .u').style.width = (100 * u).toFixed(2) + '%';
  $('#ptext').textContent = `Trusted ${(100 * t).toFixed(1)}% · Untrusted ${(100 * u).toFixed(1)}% · Unreviewed ${(100 * (1 - t - u)).toFixed(1)}% | Done ${p.tracks_done}/${p.tracks_total} tracks | ${p.objects_total} objects`;
  $('#ptext').title = `${p.total_rows} object-frames: ${p.trusted} trusted, ${p.untrusted} untrusted, ${p.unreviewed} unreviewed; ${p.linked_groups} linked groups`;
}
function markSaved() { $('#saved').textContent = 'Saved ' + new Date().toLocaleTimeString(); }
function applyState(r) {
  if (r.progress) setProgress(r.progress);
  if ('can_undo' in r) $('#btn-undo').disabled = !r.can_undo;
  if (r.tracks) for (const [id, v] of Object.entries(r.tracks)) Object.assign(S.byId.get(+id), v);
  if (r.suggestions) { S.suggestions = r.suggestions; renderSuggestions(); }
  S.obj.clear();
  markSaved();
  renderList();
}

/* ---------------- left: tracks ---------------- */
function segGradient(t) {
  const len = t.end - t.start + 1;
  if (!t.segs || !t.segs.length) return COL.unreviewed;
  const stops = [];
  let cur = t.start;
  const push = (a, b, c) => stops.push(`${c} ${(100 * (a - t.start) / len).toFixed(2)}%`, `${c} ${(100 * (b + 1 - t.start) / len).toFixed(2)}%`);
  for (const s of t.segs) {
    if (s.start > cur) push(cur, s.start - 1, COL.unreviewed);
    push(s.start, s.end, COL[s.status]);
    cur = s.end + 1;
  }
  if (cur <= t.end) push(cur, t.end, COL.unreviewed);
  return `linear-gradient(90deg, ${stops.join(', ')})`;
}
function trackState(t) {
  const covered = (t.segs || []).reduce((n, s) => n + s.end - s.start + 1, 0);
  return { todo: covered < t.end - t.start + 1, bad: (t.segs || []).some((s) => s.status === 'untrusted') };
}
/* One entry per physical object: linked IDs are combined, unlinked IDs stand alone. */
function buildObjects() {
  const by = new Map();
  for (const t of S.tracks) {
    if (!by.has(t.obj)) by.set(t.obj, []);
    by.get(t.obj).push(t);
  }
  const out = [];
  for (const [obj, ms] of by) {
    ms.sort((a, b) => a.start - b.start);
    const n = ms.reduce((s, m) => s + m.n, 0);
    out.push({
      obj, members: ms, id: obj, start: Math.min(...ms.map((m) => m.start)), end: Math.max(...ms.map((m) => m.end)), n,
      area_med: ms.reduce((s, m) => s + m.area_med * m.n, 0) / n, wall_frac: ms.reduce((s, m) => s + m.wall_frac * m.n, 0) / n,
    });
  }
  return out;
}
function objGradient(o) {  // review state over the object's whole time span; gaps between IDs are left dark
  const len = o.end - o.start + 1, gap = COL['panel-2'], stops = [];
  let cur = o.start;
  const push = (a, b, c) => stops.push(`${c} ${(100 * (a - o.start) / len).toFixed(2)}%`, `${c} ${(100 * (b + 1 - o.start) / len).toFixed(2)}%`);
  for (const m of o.members) {
    if (m.start > cur) push(cur, m.start - 1, gap);
    let c2 = m.start;
    for (const s of m.segs || []) {
      if (s.start > c2) push(c2, s.start - 1, COL.unreviewed);
      push(s.start, s.end, COL[s.status]);
      c2 = s.end + 1;
    }
    if (c2 <= m.end) push(c2, m.end, COL.unreviewed);
    cur = Math.max(cur, m.end + 1);
  }
  return `linear-gradient(90deg, ${stops.join(', ')})`;
}
function rowHtml(label, x, extra) {
  return `<span class="id">${label}</span>
    <span class="meta"><span>${(x.start / S.fps).toFixed(2)}–${(x.end / S.fps).toFixed(2)} s</span><span>${(x.n / S.fps).toFixed(2)} s</span><span>${Math.round(x.area_med)} px</span>${extra || ''}</span>`;
}
function renderList() {
  const q = $('#f-id').value.trim(), dmin = +$('#f-dur').value || 0, amin = +$('#f-area').value || 0;
  const st = $('#f-status').value, sort = $('#f-sort').value, byObj = S.view === 'object';
  const key = { start: (t) => t.start, id: (t) => t.id, dur: (t) => -t.n, area: (t) => -t.area_med, wall: (t) => -t.wall_frac }[sort];
  const keep = (x, members) => {
    const ss = members.map(trackState);
    return (!q || members.some((m) => String(m.id).startsWith(q))) && (x.n / S.fps) >= dmin && x.area_med >= amin &&
      (st === 'all' || (st === 'todo' && ss.some((s) => s.todo)) || (st === 'bad' && ss.some((s) => s.bad)) ||
       (st === 'done' && ss.every((s) => !s.todo)) || (st === 'linked' && (members.length > 1 || members[0].gsize > 1)));
  };
  const frag = document.createDocumentFragment();
  const nObj = new Set(S.tracks.map((t) => t.obj)).size;
  $('#n-tracks').textContent = byObj ? `(${nObj})` : `(${S.tracks.length})`;
  $('#tab-tracks').firstChild.textContent = byObj ? 'Objects ' : 'Tracks ';
  if (byObj) {
    const L = buildObjects().filter((o) => keep(o, o.members)).sort((a, b) => key(a) - key(b) || a.obj - b.obj);
    S.visible = L.map((o) => o.members[0].id);
    $('#f-count').textContent = `Showing ${L.length} / ${nObj} objects (${S.tracks.length} IDs)`;
    const selObj = S.detail ? S.detail.obj : null;
    for (const o of L) {
      const multi = o.members.length > 1, open = multi && S.expanded.has(o.obj);
      const li = document.createElement('li');
      li.dataset.obj = o.obj;
      li.className = 'objrow' + (o.obj === selObj ? ' sel' : '');
      const tog = multi ? `<button type="button" class="tog" aria-expanded="${open}" title="Show the IDs of this object">${open ? '▾' : '▸'} ${o.members.length} IDs</button>` : '';
      li.innerHTML = rowHtml(multi ? `obj #${o.obj}` : `#${o.obj}`, o, tog) + `<span class="bar" style="background:${objGradient(o)}"></span>`;
      li.addEventListener('click', (e) => {
        if (e.target.closest('.tog')) {
          if (open) S.expanded.delete(o.obj); else S.expanded.add(o.obj);
          renderList(); return;
        }
        selectTrack(o.members[0].id);
      });
      frag.appendChild(li);
      if (open) for (const m of o.members) {
        const mi = document.createElement('li');
        mi.dataset.id = m.id;
        mi.className = 'member' + (m.id === S.sel ? ' sel' : '');
        mi.innerHTML = rowHtml(`#${m.id}`, m) + `<span class="bar" style="background:${segGradient(m)}"></span>`;
        mi.addEventListener('click', () => selectTrack(m.id));
        frag.appendChild(mi);
      }
    }
  } else {
    const L = S.tracks.filter((t) => keep(t, [t])).sort((a, b) => key(a) - key(b) || a.id - b.id);
    S.visible = L.map((t) => t.id);
    $('#f-count').textContent = `Showing ${L.length} / ${S.tracks.length} IDs`;
    for (const t of L) {
      const li = document.createElement('li');
      li.dataset.id = t.id;
      if (t.id === S.sel) li.className = 'sel';
      const badge = t.gsize > 1 ? `<span class="badge">object #${t.obj} · ${t.gsize} IDs</span>` : '';
      li.innerHTML = rowHtml(`#${t.id}`, t, badge) + `<span class="bar" style="background:${segGradient(t)}"></span>`;
      li.addEventListener('click', () => selectTrack(t.id));
      frag.appendChild(li);
    }
  }
  $('#tracks').replaceChildren(frag);
}
function markSelected() {
  document.querySelectorAll('#tracks li.sel').forEach((li) => li.classList.remove('sel'));
  let li = null;
  if (S.view === 'object' && S.detail) {
    const orow = $(`#tracks li.objrow[data-obj="${S.detail.obj}"]`);
    if (orow) { orow.classList.add('sel'); li = orow; }
    const m = $(`#tracks li.member[data-id="${S.sel}"]`);
    if (m) { m.classList.add('sel'); li = m; }
  } else {
    li = $(`#tracks li[data-id="${S.sel}"]`);
    if (li) li.classList.add('sel');
  }
  if (li) li.scrollIntoView({ block: 'nearest' });
}
function setView(v) {
  S.view = v;
  try { localStorage.setItem('review-view', v); } catch (e) { /* storage may be unavailable */ }
  $('#v-object').setAttribute('aria-pressed', v === 'object');
  $('#v-id').setAttribute('aria-pressed', v === 'id');
  $('#f-status option[value="linked"]').textContent = v === 'object' ? 'Made of several IDs' : 'Linked to other IDs';
  renderList(); markSelected();
}

/* ---------------- left: suggestions ---------------- */
function suggText(s) {
  return `gap ${s.gap} frames (${(s.gap / S.fps).toFixed(2)} s) · ${s.dist} px apart · area ${s.area_a} → ${s.area_b} px`;
}
function renderSuggestions() {
  $('#n-sugg').textContent = S.suggestions.length;
  $('#n-sugg').hidden = !S.suggestions.length;
  const ol = $('#sugg');
  if (!S.suggestions.length) { ol.innerHTML = '<li class="muted">No suggestions left.</li>'; return; }
  ol.innerHTML = S.suggestions.map((s, i) => `<li data-i="${i}" class="${S.compare && S.sel === s.a && S.compare.id === s.b ? 'cur' : ''}">
      <span class="pair">#${s.a} → #${s.b}</span><span class="why">${suggText(s)}</span>
      <span class="ops"><button type="button" data-op="show">Show</button><button type="button" data-op="link">Link</button><button type="button" data-op="no">Not same</button></span></li>`).join('');
  ol.querySelectorAll('button').forEach((b) => b.addEventListener('click', (e) => {
    const s = S.suggestions[+e.target.closest('li').dataset.i];
    suggAction(s, e.target.dataset.op);
  }));
}
async function suggAction(s, op) {
  if (op === 'show') return showSuggestion(s);
  try {
    if (op === 'link') {
      const r = await api('/api/link', { ids: [s.a, s.b] });
      applyState(r);
      hint(`Linked #${s.a} and #${s.b} as one object`);
      if (S.sel != null) await selectTrack(S.sel, { seek: false, keepRange: true });
    } else {
      const r = await api('/api/dismiss', { a: s.a, b: s.b });
      applyState(r);
      hint(`Marked #${s.a} and #${s.b} as different objects`);
      if (S.compare) { S.compare = null; redraw(); drawTimeline(); }
    }
  } catch (e) { hint(e.message, 4000); }
}
async function showSuggestion(s) {
  await selectTrack(s.a, { seek: false });
  const b = S.byId.get(s.b);
  const d = await api('/api/track/' + s.b);
  S.compare = { id: s.b, start: b.start, end: b.end, frames: d.frames, cx: d.cx, cy: d.cy, sugg: s };
  renderSuggestions(); renderPanel();
  await seek(Math.max(S.detail.summary.start, S.detail.summary.end - 30));
}

/* ---------------- selecting a track ---------------- */
function groupSpan() {
  const g = S.detail.group;
  return [Math.min(...g.map((m) => m.start)), Math.max(...g.map((m) => m.end))];
}
function rangeLimits() {
  const s = S.detail.summary;
  return (S.detail.group.length > 1 && $('#obj-scope').checked) ? groupSpan() : [s.start, s.end];
}
async function selectTrack(id, opts = {}) {
  const d = await api('/api/track/' + id);
  if (!opts.keepCompare) S.compare = null;
  const sameObj = S.detail && S.detail.obj === d.obj;
  S.sel = id; S.detail = d;
  if (!(opts.keepRange && sameObj)) { S.inF = d.summary.start; S.outF = d.summary.end; }
  markSelected();
  renderPanel(); renderSuggestions();
  if (opts.seek !== false) await seek(d.summary.start);
  else { redraw(); drawTimeline(); }
}
function objPath() {  // positions of the whole object (all linked IDs), sorted by frame
  const pts = [];
  for (const m of S.detail.group) for (let i = 0; i < m.frames.length; i++) pts.push([m.frames[i], m.cx[i], m.cy[i]]);
  if (S.compare) for (let i = 0; i < S.compare.frames.length; i++) pts.push([S.compare.frames[i], S.compare.cx[i], S.compare.cy[i]]);
  return pts.sort((a, b) => a[0] - b[0]);
}
function posAt(f) {
  const P = S.pathCache;
  if (!P || !P.length) return null;
  let lo = 0, hi = P.length - 1;
  if (f <= P[0][0]) return { x: P[0][1], y: P[0][2] };
  if (f >= P[hi][0]) return { x: P[hi][1], y: P[hi][2] };
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (P[m][0] <= f) lo = m; else hi = m - 1; }
  return { x: P[lo][1], y: P[lo][2] };
}

/* ---------------- image ---------------- */
async function seek(f) {
  f = clamp(Math.round(f), 0, S.N - 1);
  S.frame = f;
  $('#clock').textContent = `Frame ${f} / ${S.N - 1} · ${tstr(f)} s`;
  drawTimeline();
  try {
    const [img, objs] = await Promise.all([getImage(f), getObjects(f)]);
    if (S.frame !== f) return;
    S.lastImg = img; S.lastObjs = objs;
    redraw();
  } catch (e) { hint(`Could not load frame ${f}: ${e.message}`); }
}
function label(ctx, txt, x, y, color) {
  ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(0,0,0,0.75)'; ctx.strokeText(txt, x, y);
  ctx.fillStyle = color; ctx.fillText(txt, x, y);
}
function redraw() {
  const img = S.lastImg, objs = S.lastObjs;
  if (!img) return;
  S.pathCache = S.detail ? objPath() : null;
  const cv = $('#view'), ctx = cv.getContext('2d');
  const cw = cv.width, ch = cv.height, sc = S.meta.scale;
  let sx = 0, sy = 0, sw = cw, sh = ch;
  if (S.follow && S.detail) {
    const p = posAt(S.frame);
    if (p) { sw = cw / S.zoom; sh = ch / S.zoom; sx = clamp(p.x * sc - sw / 2, 0, cw - sw); sy = clamp(p.y * sc - sh / 2, 0, ch - sh); }
  }
  ctx.setLineDash([]);
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, cw, ch);
  const k = cw / sw;
  const T = (x, y) => [(x * sc - sx) * k, (y * sc - sy) * k];
  const showAll = $('#t-all').checked;
  const myObj = S.detail ? S.detail.obj : null;
  S.hit = [];
  ctx.font = `600 ${Math.round(15 * Math.max(1, cw / 1280))}px ui-monospace, Consolas, monospace`;
  ctx.lineJoin = 'round';
  const special = [];
  for (const o of objs) {
    const path = new Path2D();
    if (o.c && o.c.length >= 4) {
      for (let i = 0; i < o.c.length; i += 2) { const [X, Y] = T(o.c[i], o.c[i + 1]); if (i) path.lineTo(X, Y); else path.moveTo(X, Y); }
      path.closePath();
    } else {
      const [X, Y] = T(o.cx, o.cy);
      path.ellipse(X, Y, Math.max(2, o.maj / 2 * sc * k), Math.max(2, o.mi / 2 * sc * k), -o.ang * Math.PI / 180, 0, 2 * Math.PI);
    }
    const [cx, cy] = T(o.cx, o.cy);
    S.hit.push({ id: o.id, path, a: o.a, cx, cy });
    const kind = o.id === S.sel ? 'sel' : (myObj != null && o.g === myObj) ? 'grp' : (S.compare && o.id === S.compare.id) ? 'cmp' : null;
    if (kind) { special.push({ o, path, cx, cy, kind }); continue; }
    if (!showAll) continue;
    const c = o.s === 'unreviewed' ? '#f2f4f5' : COL[o.s];
    ctx.lineWidth = 3.5; ctx.strokeStyle = 'rgba(0,0,0,0.55)'; ctx.stroke(path);
    ctx.lineWidth = 1.6; ctx.strokeStyle = c; ctx.stroke(path);
    label(ctx, '#' + o.id, cx + 6, cy - 6, c);
  }
  if (S.detail) {  // trail of the whole object over the last 1.5 s
    const pts = S.pathCache.filter((p) => p[0] <= S.frame && p[0] >= S.frame - 90).map((p) => T(p[1], p[2]));
    if (pts.length > 1) {
      ctx.beginPath(); pts.forEach(([X, Y], i) => (i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y)));
      ctx.lineWidth = 2; ctx.strokeStyle = COL.accent; ctx.globalAlpha = 0.8; ctx.stroke(); ctx.globalAlpha = 1;
    }
  }
  for (const { o, path, cx, cy, kind } of special) {
    const color = kind === 'cmp' ? COL.split : COL.accent;
    ctx.setLineDash(kind === 'sel' ? [] : [7, 5]);
    ctx.lineWidth = 6; ctx.strokeStyle = 'rgba(0,0,0,0.6)'; ctx.stroke(path);
    ctx.lineWidth = 3; ctx.strokeStyle = color; ctx.stroke(path);
    ctx.setLineDash([]);
    label(ctx, kind === 'sel' ? `#${o.id}` : kind === 'grp' ? `#${o.id} (same object)` : `#${o.id} (suggested)`, cx + 8, cy - 8, color);
  }
  if (S.detail && !special.length) {
    const [a, b] = groupSpan();
    const name = S.detail.group.length > 1 ? `Object #${S.detail.obj}` : `#${S.sel}`;
    hintSticky(S.frame < a ? `${name} has not appeared yet (first frame ${a})` : S.frame > b ? `${name} has ended (last frame ${b})` : `${name} is not detected in this frame`);
  } else hintSticky('');
}
function canvasPick(ev) {
  const cv = $('#view'), r = cv.getBoundingClientRect();
  const x = (ev.clientX - r.left) * cv.width / r.width, y = (ev.clientY - r.top) * cv.height / r.height;
  const ctx = cv.getContext('2d');
  let best = null;
  for (const h of S.hit) if (ctx.isPointInPath(h.path, x, y) && (!best || h.a < best.a)) best = h;
  if (!best) {  // small objects are hard to hit: take the nearest within 12 screen px
    let bd = 12 * cv.width / r.width;
    for (const h of S.hit) { const d = Math.hypot(h.cx - x, h.cy - y); if (d < bd) { bd = d; best = h; } }
  }
  if (best && best.id !== S.sel) selectTrack(best.id, { seek: false });
}

/* ---------------- timeline ---------------- */
const TL = { L: 12, R: 12, H: 104 };
function tlRange() {
  if (S.tlMode === 'track' && S.detail) {
    let [a, b] = groupSpan();
    if (S.compare) { a = Math.min(a, S.compare.start); b = Math.max(b, S.compare.end); }
    const m = Math.max(15, Math.round((b - a + 1) * 0.06));
    return [Math.max(0, a - m), Math.min(S.N - 1, b + m)];
  }
  return [0, S.N - 1];
}
function tlX(f, w) { const [a, b] = tlRange(); return TL.L + (f - a) / Math.max(1, b - a) * (w - TL.L - TL.R); }
function tlF(x, w) { const [a, b] = tlRange(); return clamp(Math.round(a + (x - TL.L) / (w - TL.L - TL.R) * (b - a)), 0, S.N - 1); }
function drawTimeline() {
  const cv = $('#timeline'), dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = TL.H;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const X = (f) => tlX(f, w);
  const [a, b] = tlRange();
  const span = (b - a) / S.fps;
  const step = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10].find((s) => span / s <= 10) || 20;
  ctx.font = '11px ui-monospace, Consolas, monospace'; ctx.fillStyle = COL['ink-2']; ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
  for (let t = Math.ceil(a / S.fps / step) * step; t <= b / S.fps + 1e-9; t += step) {
    const x = X(t * S.fps);
    ctx.beginPath(); ctx.moveTo(x, 72); ctx.lineTo(x, 78); ctx.stroke();
    ctx.fillText(t.toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0) + ' s', x + 2, 92);
  }
  ctx.beginPath(); ctx.moveTo(TL.L, 72); ctx.lineTo(w - TL.R, 72); ctx.stroke();
  if (S.detail) {
    const d = S.detail;
    const bar = (start, end, segs, y, hgt) => {
      ctx.fillStyle = COL.unreviewed; ctx.fillRect(X(start), y, Math.max(2, X(end + 1) - X(start)), hgt);
      for (const s of segs) { ctx.fillStyle = COL[s.status]; ctx.fillRect(X(s.start), y, Math.max(2, X(s.end + 1) - X(s.start)), hgt); }
    };
    ctx.font = '10px ui-monospace, Consolas, monospace';
    for (const m of d.group) {  // all IDs of this object on one lane
      bar(m.start, m.end, m.segments, 34, 18);
      const me = m.id === S.sel;
      ctx.setLineDash(me ? [] : [4, 3]); ctx.lineWidth = me ? 2 : 1.2;
      ctx.strokeStyle = me ? COL.accent : COL.ink;
      ctx.strokeRect(X(m.start), 34, Math.max(2, X(m.end + 1) - X(m.start)), 18);
      ctx.setLineDash([]);
      if (X(m.end + 1) - X(m.start) > 22 || d.group.length > 1) { ctx.fillStyle = me ? COL.accent : COL.ink; ctx.fillText('#' + m.id, X(m.start) + 2, 31); }
    }
    if (S.compare) {
      const c = S.compare, cb = S.byId.get(c.id);
      bar(c.start, c.end, cb.segs || [], 34, 18);
      ctx.setLineDash([4, 3]); ctx.lineWidth = 1.5; ctx.strokeStyle = COL.split;
      ctx.strokeRect(X(c.start), 34, Math.max(2, X(c.end + 1) - X(c.start)), 18); ctx.setLineDash([]);
      ctx.fillStyle = COL.split; ctx.fillText(`#${c.id} (suggested)`, X(c.start) + 2, 31);
    }
    ctx.fillStyle = COL.wall;
    for (const [p, q] of d.wall_spans) ctx.fillRect(X(p), 56, Math.max(1, X(q + 1) - X(p)), 5);
    ctx.fillStyle = COL.flag;
    for (const f of d.flags) ctx.fillRect(X(f.frame) - 1, 6, 3, 14);
    for (const e of d.events) {
      const x = X(e.frame);
      ctx.beginPath();
      if (e.type === 'merge') { ctx.fillStyle = COL.merge; ctx.moveTo(x - 6, 6); ctx.lineTo(x + 6, 6); ctx.lineTo(x, 20); }
      else if (e.type === 'split') { ctx.fillStyle = COL.split; ctx.moveTo(x - 6, 20); ctx.lineTo(x + 6, 20); ctx.lineTo(x, 6); }
      else { ctx.fillStyle = COL['ink-2']; ctx.arc(x, 13, 3.5, 0, 2 * Math.PI); }
      ctx.fill();
    }
    if (S.inF != null && S.outF != null) {
      const x0 = X(S.inF), x1 = X(S.outF + 1);
      ctx.fillStyle = 'rgba(255,210,63,0.22)'; ctx.fillRect(x0, 30, x1 - x0, 36);
      ctx.strokeStyle = COL.accent; ctx.lineWidth = 1.5; ctx.strokeRect(x0, 30, x1 - x0, 36);
    }
  }
  const xc = X(S.frame);
  ctx.strokeStyle = COL.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xc, 2); ctx.lineTo(xc, 80); ctx.stroke();
}
function bindTimeline() {
  const cv = $('#timeline');
  let down = null, dragging = false;
  cv.addEventListener('pointerdown', (e) => {
    const r = cv.getBoundingClientRect();
    down = { x: e.clientX - r.left, f: tlF(e.clientX - r.left, r.width) }; dragging = false;
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener('pointermove', (e) => {
    if (!down) return;
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left;
    if (!dragging && Math.abs(x - down.x) > 4) dragging = true;
    if (dragging && S.detail) {
      const f = tlF(x, r.width), [lo, hi] = rangeLimits();
      setRange(clamp(Math.min(down.f, f), lo, hi), clamp(Math.max(down.f, f), lo, hi));
    }
  });
  cv.addEventListener('pointerup', () => { if (down && !dragging) seek(down.f); down = null; dragging = false; });
  window.addEventListener('resize', drawTimeline);
}

/* ---------------- right panel ---------------- */
function setRange(a, b) {
  S.inF = a; S.outF = b;
  $('#r-start').value = a; $('#r-end').value = b;
  $('#r-sec').textContent = `${tstr(a)} – ${tstr(b)} s, ${b - a + 1} frames`;
  drawTimeline();
}
function renderPanel() {
  const d = S.detail;
  $('#empty').hidden = !!d; $('#detail').hidden = !d;
  if (!d) return;
  const s = d.summary, linked = d.group.length > 1;
  $('#d-title').textContent = linked ? `Track #${s.id} · object #${d.obj}` : `Track #${s.id}`;
  const evs = d.events.filter((e) => e.type === 'merge' || e.type === 'split');
  const evHtml = evs.length ? evs.map((e) => {
    const other = e.ids.filter((i) => i !== s.id).map((i) => '#' + i).join(', ');
    const txt = e.type === 'merge' ? (e.result === s.id ? `absorbed ${other}` : `merged into #${e.result}`) : (e.ids[0] === s.id ? `split off ${other}` : `split off from #${e.ids[0]}`);
    return `<a data-f="${e.frame}">${tstr(e.frame)} s ${txt}</a>`;
  }).join('<br>') : 'none';
  $('#d-facts').innerHTML = `
    <dt>Frames</dt><dd>${s.start} – ${s.end} (${s.n} frames)</dd>
    <dt>Time</dt><dd>${tstr(s.start)} – ${tstr(s.end)} s</dd>
    <dt>Area</dt><dd>median ${Math.round(s.area_med)} · max ${Math.round(s.area_max)} px</dd>
    <dt>Distance moved</dt><dd>${Math.round(s.path)} px</dd>
    <dt>Touching wall/debris</dt><dd>${(100 * s.wall_frac).toFixed(0)}% of frames</dd>
    <dt>Merges &amp; splits</dt><dd>${evHtml}</dd>`;
  $('#d-facts').querySelectorAll('a[data-f]').forEach((a) => a.addEventListener('click', () => seek(+a.dataset.f)));
  // same object
  const grp = $('#grp');
  grp.innerHTML = linked
    ? d.group.map((m) => `<span class="chip ${m.id === s.id ? 'me' : ''}" data-id="${m.id}">#${m.id} <span class="muted">${tstr(m.start)}–${tstr(m.end)} s</span>${m.id === s.id ? `<button type="button" data-unlink="${m.id}" title="Remove this ID from the object">Unlink</button>` : ''}</span>`).join('')
    : '<span class="muted">This ID is not linked to any other ID.</span>';
  grp.querySelectorAll('.chip').forEach((c) => c.addEventListener('click', (e) => {
    if (e.target.dataset.unlink) return unlink(+e.target.dataset.unlink);
    if (+c.dataset.id !== S.sel) selectTrack(+c.dataset.id, { keepRange: true });
  }));
  const ds = $('#d-sugg');
  const mine = d.suggestions;
  ds.innerHTML = mine.map((x, i) => `<li data-i="${i}"><span>Possibly the same object: <b>#${x.a} → #${x.b}</b></span><span class="why">${suggText(x)}</span>
      <span class="ops"><button type="button" data-op="show">Show</button><button type="button" data-op="link">Link</button><button type="button" data-op="no">Not same</button></span></li>`).join('');
  ds.querySelectorAll('button').forEach((bn) => bn.addEventListener('click', (e) => suggAction(mine[+e.target.closest('li').dataset.i], e.target.dataset.op)));
  $('#r-obj').hidden = !linked; $('#obj-scope-wrap').hidden = !linked;
  setRange(S.inF, S.outF);
  renderSegs();
  const fl = [...d.flags.map((f) => ({ frame: f.frame, k: f.kind, text: f.text })),
    ...evs.map((e) => ({ frame: e.frame, k: e.type, text: `${EVENT[e.type]}: ${e.ids.map((i) => '#' + i).join(' + ')}` })),
    ...d.wall_spans.map(([p, q]) => ({ frame: p, k: 'wall', text: `Starts touching wall/debris (until frame ${q})` }))]
    .sort((x, y) => x.frame - y.frame);
  $('#flags').innerHTML = fl.length ? fl.map((f) => `<li data-f="${f.frame}"><span class="k ${f.k}"></span>${tstr(f.frame)} s · ${f.text}</li>`).join('') : '<li class="muted">none</li>';
  $('#flags').querySelectorAll('li[data-f]').forEach((li) => li.addEventListener('click', () => seek(+li.dataset.f)));
}
function renderSegs() {
  const d = S.detail, ul = $('#segs');
  const rows = [];
  for (const m of d.group) for (const s of m.segments) rows.push({ ...s, track: m.id });
  if (!rows.length) { ul.innerHTML = `<li class="muted" style="border:0;background:none;padding:0">Nothing marked yet; the whole ${d.group.length > 1 ? 'object' : 'track'} is unreviewed.</li>`; return; }
  ul.innerHTML = rows.map((s, i) => `<li class="${s.status}">
      <span class="when">${d.group.length > 1 ? `#${s.track} · ` : ''}frames ${s.start}–${s.end} (${tstr(s.start)}–${tstr(s.end)} s) · ${LABEL[s.status]}</span>
      <span class="ops"><button type="button" data-i="${i}" data-op="pick">Select</button><button type="button" data-i="${i}" data-op="clear">Clear</button></span>
      <span class="why">${[s.reason, s.note].filter(Boolean).join(' · ') || '&nbsp;'}</span></li>`).join('');
  ul.querySelectorAll('button').forEach((b) => b.addEventListener('click', async () => {
    const s = rows[+b.dataset.i];
    if (s.track !== S.sel) await selectTrack(s.track, { seek: false, keepRange: true });
    if (b.dataset.op === 'pick') { setRange(s.start, s.end); seek(s.start); }
    else apply('unreviewed', s.start, s.end, false);
  }));
}
async function apply(status, a = S.inF, b = S.outF, wholeObject = null) {
  if (!S.detail) { hint('Select a track first'); return; }
  if (a == null || b == null || a > b) { hint('Select a range first'); return; }
  const whole = wholeObject ?? (S.detail.group.length > 1 && $('#obj-scope').checked);
  try {
    const r = await api('/api/set', {
      track: S.sel, start: a, end: b, status, whole_object: whole,
      reason: status === 'untrusted' ? $('#reason').value : '', note: status === 'unreviewed' ? '' : $('#note').value.trim(),
    });
    applyState(r);
    await selectTrack(S.sel, { seek: false, keepRange: true, keepCompare: true });
    getObjects(S.frame).then((o) => { S.lastObjs = o; redraw(); });
    hint(`${r.touched.map((t) => '#' + t).join(', ')} frames ${a}–${b} → ${LABEL[status]}`);
  } catch (e) { hint('Not saved: ' + e.message, 4000); }
}
async function linkTyped() {
  const ids = ($('#link-ids').value.match(/\d+/g) || []).map(Number);
  if (!S.detail || !ids.length) { hint('Type one or more IDs, e.g. 107, 111'); return; }
  try {
    const r = await api('/api/link', { ids: [S.sel, ...ids] });
    applyState(r);
    $('#link-ids').value = '';
    hint(`Linked ${r.touched.map((t) => '#' + t).join(', ')} as one object`);
    await selectTrack(S.sel, { seek: false, keepRange: true });
  } catch (e) { hint(e.message, 4500); }
}
async function unlink(id) {
  try {
    const r = await api('/api/unlink', { id });
    applyState(r);
    hint(`#${id} is no longer linked to the other IDs`);
    await selectTrack(S.sel, { seek: false, keepRange: true });
  } catch (e) { hint(e.message, 4000); }
}
async function undo() {
  try {
    const r = await api('/api/undo', {});
    applyState(r);
    if (!r.undone) { hint('Nothing to undo'); return; }
    hint(`Undid the last change (${r.undone.action}) on ${r.undone.tracks.map((t) => '#' + t).join(', ')}`);
    if (S.sel != null) await selectTrack(S.sel, { seek: false, keepRange: true, keepCompare: true });
    getObjects(S.frame).then((o) => { S.lastObjs = o; redraw(); });
  } catch (e) { hint('Undo failed: ' + e.message, 4000); }
}

/* ---------------- playback ---------------- */
function setPlaying(on) {
  S.playing = on;
  $('#b-play').textContent = on ? 'Pause' : 'Play';
  if (on) loop();
}
async function loop() {
  while (S.playing) {
    const t0 = performance.now();
    let next = S.frame + 1;
    const lim = $('#t-loop').checked && S.detail && S.inF != null ? [S.inF, S.outF] : null;
    if (lim && (next > lim[1] || next < lim[0])) next = lim[0];
    if (!lim && next >= S.N) { setPlaying(false); break; }
    prefetch(next + 1, 30);
    await seek(next);
    const wait = 1000 / (S.fps * S.speed) - (performance.now() - t0);
    if (wait > 0) await sleep(wait);
  }
}

/* ---------------- export ---------------- */
async function doExport(video) {
  $('#dlg-title').textContent = video ? 'Make check video' : 'Export & check';
  $('#dlg-body').textContent = video ? 'Writing the colour-coded check video. A 22 s video takes about 30 s…' : 'Exporting…';
  $('#dialog').hidden = false;
  try {
    const r = await api('/api/export', { video });
    $('#dlg-body').textContent = r.report + (r.video ? `\nCheck video: ${r.video}\n(green = trusted, red = untrusted, grey = unreviewed; label #object(track))` : '') +
      '\n\nFiles in the result folder: trusted.csv, untrusted.csv, unreviewed.csv, objects_reviewed.csv, events_reviewed.csv, review.json';
    markSaved();
  } catch (e) { $('#dlg-body').textContent = 'Export failed: ' + e.message; }
}

/* ---------------- keyboard ---------------- */
function step(d) {
  const L = S.visible || [];
  if (!L.length) return;
  let i = L.indexOf(S.sel);
  if (S.view === 'object' && S.detail) i = L.findIndex((id) => S.byId.get(id).obj === S.detail.obj);
  selectTrack(L[clamp(i < 0 ? 0 : i + d, 0, L.length - 1)]);
}
function bindKeys() {
  document.addEventListener('keydown', (e) => {
    const tag = (e.target.tagName || '').toLowerCase();
    if ((tag === 'input' || tag === 'select' || tag === 'textarea') && e.key !== 'Escape') return;
    if (!$('#dialog').hidden) { if (e.key === 'Escape' || e.key === 'Enter') $('#dialog').hidden = true; return; }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const tr = S.detail && S.detail.summary;
    const lim = () => rangeLimits();
    const m = {
      ' ': () => setPlaying(!S.playing),
      ArrowLeft: () => seek(S.frame - (e.shiftKey ? 10 : 1)),
      ArrowRight: () => seek(S.frame + (e.shiftKey ? 10 : 1)),
      Home: () => tr && seek(tr.start), End: () => tr && seek(tr.end),
      i: () => { if (!tr) return; const [lo, hi] = lim(); const f = clamp(S.frame, lo, hi); setRange(f, Math.max(f, S.outF ?? hi)); },
      o: () => { if (!tr) return; const [lo, hi] = lim(); const f = clamp(S.frame, lo, hi); setRange(Math.min(S.inF ?? lo, f), f); },
      a: () => tr && setRange(tr.start, tr.end),
      t: () => apply('trusted'), u: () => apply('untrusted'), c: () => apply('unreviewed'),
      '[': () => step(-1), ']': () => step(1),
      z: () => { $('#t-follow').checked = !$('#t-follow').checked; S.follow = $('#t-follow').checked; redraw(); },
    };
    const f = m[e.key] || m[e.key.toLowerCase()];
    if (f) { e.preventDefault(); f(); }
  });
}

/* ---------------- start ---------------- */
function setTab(tab) {
  S.tab = tab;
  $('#tab-tracks').setAttribute('aria-selected', tab === 'tracks');
  $('#tab-sugg').setAttribute('aria-selected', tab === 'sugg');
  $('#pane-tracks').hidden = tab !== 'tracks';
  $('#pane-sugg').hidden = tab !== 'sugg';
}
async function init() {
  const m = await api('/api/meta');
  S.meta = m; S.N = m.n_frames; S.fps = m.fps;
  S.tracks = m.tracks; S.tracks.forEach((t) => S.byId.set(t.id, t));
  S.suggestions = m.suggestions;
  const short = m.result.split(/[\\/]/).filter(Boolean).slice(-2).join('/');
  $('#src').textContent = `${m.video} · ${short}` + (m.has_contours ? '' : ' · no contours.npz, objects drawn as ellipses');
  $('#src').title = m.result;
  const cv = $('#view'); cv.width = Math.round(m.width * m.scale); cv.height = Math.round(m.height * m.scale);
  $('#reason').innerHTML = '<option value="">(not specified)</option>' + m.reasons.map((r) => `<option>${r}</option>`).join('');
  setProgress(m.progress);
  $('#btn-undo').disabled = !m.can_undo;
  ['#f-id', '#f-dur', '#f-area', '#f-status', '#f-sort'].forEach((s) => $(s).addEventListener('input', renderList));
  let view = 'object';
  try { view = localStorage.getItem('review-view') || 'object'; } catch (e) { /* storage may be unavailable */ }
  $('#v-object').addEventListener('click', () => setView('object'));
  $('#v-id').addEventListener('click', () => setView('id'));
  setView(view === 'id' ? 'id' : 'object'); renderSuggestions();
  $('#tab-tracks').addEventListener('click', () => setTab('tracks'));
  $('#tab-sugg').addEventListener('click', () => setTab('sugg'));
  $('#view').addEventListener('click', canvasPick);
  $('#b-play').addEventListener('click', () => setPlaying(!S.playing));
  $('#b-back').addEventListener('click', () => seek(S.frame - 1));
  $('#b-fwd').addEventListener('click', () => seek(S.frame + 1));
  $('#b-back10').addEventListener('click', () => seek(S.frame - 10));
  $('#b-fwd10').addEventListener('click', () => seek(S.frame + 10));
  $('#b-start').addEventListener('click', () => S.detail && seek(S.detail.summary.start));
  $('#b-end').addEventListener('click', () => S.detail && seek(S.detail.summary.end));
  $('#speed').addEventListener('change', (e) => { S.speed = +e.target.value; });
  S.speed = +$('#speed').value;
  $('#t-follow').addEventListener('change', (e) => { S.follow = e.target.checked; redraw(); });
  $('#t-all').addEventListener('change', redraw);
  const tlBtn = (mode) => { S.tlMode = mode; $('#tl-track').setAttribute('aria-pressed', mode === 'track'); $('#tl-video').setAttribute('aria-pressed', mode === 'video'); drawTimeline(); };
  $('#tl-track').addEventListener('click', () => tlBtn('track'));
  $('#tl-video').addEventListener('click', () => tlBtn('video'));
  $('#r-start').addEventListener('change', (e) => { const [lo, hi] = rangeLimits(); const f = clamp(+e.target.value, lo, hi); setRange(f, Math.max(f, S.outF)); });
  $('#r-end').addEventListener('change', (e) => { const [lo, hi] = rangeLimits(); const f = clamp(+e.target.value, lo, hi); setRange(Math.min(S.inF, f), f); });
  $('#r-in').addEventListener('click', () => { const [lo, hi] = rangeLimits(); const f = clamp(S.frame, lo, hi); setRange(f, Math.max(f, S.outF)); });
  $('#r-out').addEventListener('click', () => { const [lo, hi] = rangeLimits(); const f = clamp(S.frame, lo, hi); setRange(Math.min(S.inF, f), f); });
  $('#r-all').addEventListener('click', () => { const s = S.detail.summary; setRange(s.start, s.end); });
  $('#r-obj').addEventListener('click', () => { const [a, b] = groupSpan(); $('#obj-scope').checked = true; setRange(a, b); });
  $('#obj-scope').addEventListener('change', () => { const [lo, hi] = rangeLimits(); setRange(clamp(S.inF, lo, hi), clamp(S.outF, lo, hi)); });
  $('#s-trusted').addEventListener('click', () => apply('trusted'));
  $('#s-untrusted').addEventListener('click', () => apply('untrusted'));
  $('#s-clear').addEventListener('click', () => apply('unreviewed'));
  $('#btn-link').addEventListener('click', linkTyped);
  $('#link-ids').addEventListener('keydown', (e) => { if (e.key === 'Enter') linkTyped(); });
  $('#btn-undo').addEventListener('click', undo);
  $('#btn-export').addEventListener('click', () => doExport(false));
  $('#btn-video').addEventListener('click', () => doExport(true));
  $('#dlg-close').addEventListener('click', () => { $('#dialog').hidden = true; });
  bindTimeline(); bindKeys();
  await seek(0);
  prefetch(1, 30);
}
init().catch((e) => { document.body.innerHTML = `<p style="padding:20px">Could not start: ${e.message}</p>`; });
