'use strict';
// Analysis check viewer: the video next to the round-2 curves. Click a curve to jump to that frame.

const PAL = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
const ROLE = { confirmed: '#46c35a', uncertain: '#ff9f1c', excluded: '#ec4f4f' };
const ACCENT = '#ffd23f';
const INK = '#e6eaed', INK2 = '#9aa5ae', GRID = '#2c3238', AXIS = '#3d454d';

const S = {
  meta: null, field: null, sel: 'field', obj: null, objCache: new Map(),
  frame: 0, playing: false, range: 'obj', hoverF: null,
  imgs: new Map(), regions: new Map(), objIndex: new Map(),
};
const $ = (id) => document.getElementById(id);
const api = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(p + ' ' + r.status); return r.json(); });
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : (typeof v === 'number' ? v.toFixed(d) : v);

// ---------------------------------------------------------------- loading
async function init() {
  S.meta = await api('/api/meta');
  S.field = await api('/api/field');
  S.fieldByFrame = new Map(S.field.rows.map((r) => [r.frame, r]));
  $('src').textContent = `${S.meta.video} · ${S.meta.result}`;
  buildList();
  buildMoments();
  const first = S.meta.objects.find((o) => o.main);
  await select(first ? first.object_id : 'field', { seek: true });
  bindUi();
  resizeChart();
}

function frameImage(f) {
  if (S.imgs.has(f)) return S.imgs.get(f);
  const p = new Promise((res) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => res(null);
    im.src = `/api/frame/${f}`;
  });
  S.imgs.set(f, p);
  if (S.imgs.size > 240) S.imgs.delete(S.imgs.keys().next().value);
  return p;
}

function frameRegions(f) {
  if (S.regions.has(f)) return S.regions.get(f);
  const p = api(`/api/regions/${f}`).catch(() => []);
  S.regions.set(f, p);
  if (S.regions.size > 240) S.regions.delete(S.regions.keys().next().value);
  return p;
}

async function objectData(oid) {
  if (!S.objCache.has(oid)) {
    const d = await api(`/api/object/${oid}`);
    d.byFrame = new Map(d.rows.map((r) => [r.frame, r]));
    S.objCache.set(oid, d);
  }
  return S.objCache.get(oid);
}

// ---------------------------------------------------------------- list
function buildList() {
  const ol = $('objects');
  const items = [];
  items.push('<li class="head">Whole field</li>');
  items.push(`<li class="item" data-sel="field"><i class="chip" style="background:${INK2}"></i>
    <span class="name">All regions</span><span class="meta">count, area, spread, clusters</span></li>`);
  const mains = S.meta.objects.filter((o) => o.main);
  const others = S.meta.objects.filter((o) => !o.main);
  const row = (o) => `<li class="item" data-sel="${o.object_id}"><i class="chip" style="background:${o.color || AXIS}"></i>
    <span class="name">#${o.object_id}${o.ids.length > 1 ? ` <span class="muted">(${o.ids.length} IDs)</span>` : ''}</span>
    <span class="meta">${o.t0.toFixed(1)}–${o.t1.toFixed(1)} s · ${o.trusted_s.toFixed(1)} s · ≤${Math.round(o.area_max)} px²</span></li>`;
  if (mains.length) items.push('<li class="head">Main objects</li>', ...mains.map(row));
  if (others.length) items.push(`<li class="head">Other objects (${others.length})</li>`, ...others.map(row));
  ol.innerHTML = items.join('');
  ol.addEventListener('click', (e) => {
    const li = e.target.closest('li.item');
    if (!li) return;
    const v = li.dataset.sel;
    select(v === 'field' ? 'field' : Number(v), { seek: true });
  });
}

function markSelected() {
  document.querySelectorAll('#objects li.item').forEach((li) => {
    li.classList.toggle('sel', li.dataset.sel === String(S.sel));
  });
  const li = document.querySelector('#objects li.sel');
  if (li) li.scrollIntoView({ block: 'nearest' });
}

function buildMoments() {
  const sel = $('moments');
  (S.meta.checks || []).forEach((c, k) => {
    const o = document.createElement('option');
    o.value = String(k);
    const who = c.kind === 'field' ? 'Whole field' : `#${c.object_id}`;
    o.textContent = `Check ${k + 1} · ${who} · ${(c.frames[1] / S.meta.fps).toFixed(2)} s — ${c.what}`;
    sel.appendChild(o);
  });
  sel.addEventListener('change', async () => {
    if (sel.value === '') return;
    const c = S.meta.checks[Number(sel.value)];
    $('t-zoom').checked = c.kind !== 'field';
    await select(c.kind === 'field' ? 'field' : c.object_id, { seek: false });
    seek(c.frames[1]);
  });
}

async function select(sel, { seek: doSeek = false } = {}) {
  S.sel = sel;
  S.obj = sel === 'field' ? null : await objectData(sel);
  markSelected();
  $('c-title').textContent = sel === 'field' ? 'Whole field' : `Object #${sel}`;
  $('c-help').textContent = sel === 'field'
    ? 'Click a curve to jump to that frame. All regions except confirmed false targets. Orange band: uncertain regions hold more than 5 % of the area. A jump at an enter / leave view event is a field-of-view effect.'
    : 'Click a curve to jump to that frame. Faint: shape pressed against a wall, or region cut by the image edge. Dashed lines: the trusted data are cut there (not joined across).';
  $('r-obj').disabled = sel === 'field';
  if (doSeek && S.obj && S.obj.rows.length && !S.obj.byFrame.has(S.frame)) seek(S.obj.rows[0].frame);
  else { drawFrame(); drawChart(); }
}

// ---------------------------------------------------------------- frame view
let drawToken = 0;
async function drawFrame() {
  const f = S.frame;
  const token = ++drawToken;
  const [im, regs] = await Promise.all([frameImage(f), frameRegions(f)]);
  if (token !== drawToken || !im) return;
  const cv = $('view');
  const sc = S.meta.scale;
  if (cv.width !== im.naturalWidth || cv.height !== im.naturalHeight) {
    cv.width = im.naturalWidth;
    cv.height = im.naturalHeight;
  }
  const g = cv.getContext('2d');
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, cv.width, cv.height);
  // zoom window around the selected object
  let zoom = 1, ox = 0, oy = 0;
  const pos = objectPosition(f);
  if ($('t-zoom').checked && pos) {
    zoom = 3;
    ox = Math.min(Math.max(0, pos[0] * sc - cv.width / (2 * zoom)), cv.width - cv.width / zoom);
    oy = Math.min(Math.max(0, pos[1] * sc - cv.height / (2 * zoom)), cv.height - cv.height / zoom);
  }
  g.setTransform(zoom, 0, 0, zoom, -ox * zoom, -oy * zoom);
  g.imageSmoothingEnabled = zoom === 1;
  g.drawImage(im, 0, 0);
  const lw = 1 / zoom;
  // outlines
  for (const r of regs) {
    if (!r.pts) continue;
    const mine = S.sel !== 'field' && r.obj === S.sel;
    if (!mine && !$('t-all').checked) continue;
    g.beginPath();
    for (let i = 0; i < r.pts.length; i += 2) {
      const x = (r.pts[i] + 0.5) * sc, y = (r.pts[i + 1] + 0.5) * sc;
      if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
    }
    g.closePath();
    if (mine) {
      g.strokeStyle = 'rgba(0,0,0,0.8)'; g.lineWidth = 4 * lw; g.stroke();
      g.strokeStyle = ACCENT; g.lineWidth = 2 * lw; g.stroke();
    } else {
      g.strokeStyle = 'rgba(0,0,0,0.6)'; g.lineWidth = 3 * lw; g.stroke();
      g.strokeStyle = ROLE[r.role] || INK2; g.lineWidth = 1.2 * lw; g.stroke();
    }
  }
  // path of the selected object (pieces are not joined across cuts)
  if (S.obj && $('t-path').checked) {
    g.strokeStyle = 'rgba(255,210,63,0.75)';
    g.lineWidth = 1.4 * lw;
    g.beginPath();
    let prev = null;
    for (const r of S.obj.rows) {
      const x = r.x_px * sc, y = r.y_px * sc;
      if (!prev || prev.segment !== r.segment || r.frame - prev.frame > 1) g.moveTo(x, y); else g.lineTo(x, y);
      prev = r;
    }
    g.stroke();
    if (pos) {
      g.fillStyle = ACCENT;
      g.beginPath(); g.arc(pos[0] * sc, pos[1] * sc, 3 * lw, 0, 2 * Math.PI); g.fill();
    }
  }
  $('stamp').textContent = `frame ${f} · ${(f / S.meta.fps).toFixed(3)} s${zoom > 1 ? ' · 3× zoom' : ''}`;
  readout();
  // prefetch while playing
  if (S.playing) for (let k = 1; k <= 8; k++) if (f + k < S.meta.n_frames) { frameImage(f + k); frameRegions(f + k); }
}

function objectPosition(f) {
  if (!S.obj || !S.obj.rows.length) return null;
  const r = S.obj.byFrame.get(f);
  if (r) return [r.x_px, r.y_px];
  // nearest trusted frame within 0.5 s keeps the zoom steady across short gaps
  let best = null;
  for (const q of S.obj.rows) if (Math.abs(q.frame - f) <= S.meta.fps / 2 && (!best || Math.abs(q.frame - f) < Math.abs(best.frame - f))) best = q;
  return best ? [best.x_px, best.y_px] : null;
}

function readout() {
  const f = S.frame;
  let t;
  if (S.sel === 'field') {
    const r = S.fieldByFrame.get(f);
    t = r ? `${r.n_regions} regions (${r.n_confirmed} trusted) · total area ${fmt(r.total_area_px2, 0)} px² · spread ${fmt(r.spread_px, 0)} px · `
      + `clusters hold ${fmt(100 * r.cluster_area_share, 0)} % · largest ${fmt(100 * r.largest_share, 0)} % · median gap ${fmt(r.nn_gap_median_px, 0)} px · ${r.field_quality}` : '';
  } else {
    const r = S.obj.byFrame.get(f);
    t = r ? `#${S.sel} (track ${r.id}, segment ${r.segment}) · x ${fmt(r.x_px)} y ${fmt(r.y_px)} px · speed ${fmt(r.speed_px_s, 0)} px/s · `
      + `area ${fmt(r.area_px2, 0)} px² · aspect ${fmt(r.aspect_ratio, 2)} · circularity ${fmt(r.circularity, 2)} · shape ${r.shape_quality}`
      + (r.at_image_edge ? ' · cut by image edge' : '')
      : `#${S.sel} has no trusted data in this frame`;
  }
  $('readout').textContent = t;
}

function seek(f) {
  S.frame = Math.max(0, Math.min(S.meta.n_frames - 1, Math.round(f)));
  drawFrame();
  drawChart();
}

// ---------------------------------------------------------------- charts
function objectPanels() {
  const c = S.obj.color || PAL[0];
  const c2 = c === PAL[1] ? PAL[0] : PAL[1];
  const wall = (r) => r.shape_quality === 'wall contact';
  const poor = (r) => r.shape_quality === 'poor';
  return [
    { title: 'Position (centroid)', unit: 'px', series: [{ k: 'x_px', name: 'x', c }, { k: 'y_px', name: 'y', c: c2 }] },
    { title: 'Speed', unit: 'px/s', zero: true, series: [{ k: 'speed_px_s', c, faint: (r) => !!r.at_image_edge }] },
    { title: 'Visible area', unit: 'px²', zero: true, series: [{ k: 'area_px2', c, faint: wall, hide: poor }] },
    { title: 'Aspect ratio', unit: '', min: 1, series: [{ k: 'aspect_ratio', c, faint: wall, hide: poor }] },
    { title: 'Circularity', unit: '', series: [{ k: 'circularity', c, faint: wall, hide: poor }] },
    { title: 'Orientation (only when elongated)', unit: 'deg', fixed: [0, 180], series: [{ k: 'orientation_deg', c, dots: true, hide: (r) => !r.orientation_ok }] },
  ];
}

function fieldPanels() {
  return [
    { title: 'Number of visible regions', unit: '', series: [{ k: 'n_regions', name: 'all', c: PAL[0] }, { k: 'n_confirmed', name: 'trusted only', c: PAL[1] }] },
    { title: 'Total visible area', unit: 'px²', series: [{ k: 'total_area_px2', c: PAL[0] }] },
    { title: 'Spread (area-weighted)', unit: 'px', series: [{ k: 'spread_px', c: PAL[0] }] },
    { title: 'Share of area: in clusters / largest region', unit: '', fixed: [0, 1], series: [{ k: 'cluster_area_share', name: 'clusters', c: PAL[0] }, { k: 'largest_share', name: 'largest', c: PAL[1] }] },
    { title: 'Median gap to the nearest region', unit: 'px', series: [{ k: 'nn_gap_median_px', c: PAL[0] }] },
    { title: 'Field events', events: true, h: 0.8 },
  ];
}

let CH = null;   // chart geometry for hit-testing

function resizeChart() {
  const cv = $('chart');
  const r = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.max(200, Math.round(r.width * dpr));
  cv.height = Math.max(200, Math.round(r.height * dpr));
  drawChart();
}

function timeRange() {
  const T = S.meta.n_frames / S.meta.fps;
  if (S.sel !== 'field' && S.range === 'obj' && S.obj && S.obj.rows.length) {
    const t0 = S.obj.rows[0].frame / S.meta.fps, t1 = S.obj.rows[S.obj.rows.length - 1].frame / S.meta.fps;
    const pad = Math.max(0.2, 0.04 * (t1 - t0));
    return [Math.max(0, t0 - pad), Math.min(T, t1 + pad)];
  }
  return [0, T];
}

function niceTicks(lo, hi, n = 3) {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function drawChart() {
  const cv = $('chart');
  if (!S.meta || !cv.width) return;
  const dpr = window.devicePixelRatio || 1;
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cv.width / dpr, H = cv.height / dpr;
  g.clearRect(0, 0, W, H);
  const field = S.sel === 'field';
  const panels = field ? fieldPanels() : objectPanels();
  const rows = field ? S.field.rows : S.obj.rows;
  const [t0, t1] = timeRange();
  const fps = S.meta.fps;
  const L = field ? 74 : 56, R = 10, B = 24;
  const weights = panels.map((p) => p.h || 1);
  const unitH = (H - B) / weights.reduce((a, b) => a + b, 0);
  const X = (t) => L + (t - t0) / (t1 - t0) * (W - L - R);
  CH = { L, R, W, t0, t1 };
  const fShow = S.hoverF !== null ? S.hoverF : S.frame;
  let y0 = 0;
  g.font = '11px ' + getComputedStyle(document.body).fontFamily;
  g.textBaseline = 'middle';
  for (const [pi, p] of panels.entries()) {
    const ph = unitH * weights[pi];
    const top = y0 + 18, bot = y0 + ph - 8;
    // title and value at the shown frame
    g.fillStyle = INK; g.textAlign = 'left';
    g.font = '600 12px ' + getComputedStyle(document.body).fontFamily;
    g.fillText(p.title + (p.unit ? ` (${p.unit})` : ''), L, y0 + 8);
    g.font = '11px ' + getComputedStyle(document.body).fontFamily;
    if (!p.events) {
      const vr = field ? S.fieldByFrame.get(fShow) : S.obj.byFrame.get(fShow);
      g.textAlign = 'right';
      let xr = W - R;
      for (const s of [...p.series].reverse()) {
        const v = vr ? vr[s.k] : null;
        const txt = `${s.name ? s.name + ' ' : ''}${v === null || v === undefined ? '–' : (Math.abs(v) < 10 ? v.toFixed(2) : v.toFixed(0))}`;
        g.fillStyle = s.c; g.fillRect(xr - g.measureText(txt).width - 12, y0 + 5, 8, 6);
        g.fillStyle = INK2; g.fillText(txt, xr, y0 + 8);
        xr -= g.measureText(txt).width + 22;
      }
    }
    if (p.events) {
      drawEvents(g, p, top, bot, X, t0, t1);
    } else {
      // y range from visible data
      let lo = Infinity, hi = -Infinity;
      for (const r of rows) {
        const t = r.frame / fps;
        if (t < t0 || t > t1) continue;
        for (const s of p.series) {
          const v = r[s.k];
          if (v === null || v === undefined || (s.hide && s.hide(r))) continue;
          lo = Math.min(lo, v); hi = Math.max(hi, v);
        }
      }
      if (p.fixed) [lo, hi] = p.fixed;
      if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
      if (p.zero) lo = Math.min(0, lo);
      if (p.min !== undefined) lo = Math.min(p.min, lo);
      if (hi - lo < 1e-9) { hi = lo + 1; }
      if (!p.fixed) { const pad = 0.06 * (hi - lo); hi += pad; if (!p.zero && p.min === undefined) lo -= pad; }
      const Y = (v) => bot - (v - lo) / (hi - lo) * (bot - top);
      // grid
      g.strokeStyle = GRID; g.lineWidth = 1; g.textAlign = 'right'; g.fillStyle = INK2;
      for (const v of niceTicks(lo, hi, 3)) {
        const y = Math.round(Y(v)) + 0.5;
        if (y < top - 1 || y > bot + 1) continue;
        g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke();
        g.fillText(Math.abs(v) >= 1000 ? (v / 1000).toFixed(v % 1000 ? 1 : 0) + 'k' : String(+v.toFixed(2)), L - 6, y);
      }
      // uncertain field frames
      if (field) shadeUncertain(g, rows, top, bot, X, t0, t1);
      // cuts between segments
      if (!field) drawCuts(g, top, bot, X, t0, t1);
      // series
      for (const s of p.series) drawSeries(g, rows, s, X, Y, t0, t1, field);
      g.strokeStyle = AXIS; g.beginPath(); g.moveTo(L, Math.round(bot) + 0.5); g.lineTo(W - R, Math.round(bot) + 0.5); g.stroke();
    }
    y0 += ph;
  }
  // time axis
  g.fillStyle = INK2; g.textAlign = 'center'; g.textBaseline = 'top';
  for (const v of niceTicks(t0, t1, 6)) g.fillText(v + ' s', X(v), H - B + 6);
  g.textBaseline = 'middle';
  // current frame and hover
  const xc = X(S.frame / fps);
  if (xc >= L && xc <= W - R) { g.strokeStyle = ACCENT; g.lineWidth = 1.5; g.beginPath(); g.moveTo(xc, 0); g.lineTo(xc, H - B); g.stroke(); }
  if (S.hoverF !== null) {
    const xh = X(S.hoverF / fps);
    g.strokeStyle = 'rgba(230,234,237,0.45)'; g.lineWidth = 1; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(xh, 0); g.lineTo(xh, H - B); g.stroke(); g.setLineDash([]);
    g.fillStyle = INK; g.textAlign = xh > W - 90 ? 'right' : 'left';
    g.fillText(`${(S.hoverF / fps).toFixed(3)} s · frame ${S.hoverF}`, xh + (xh > W - 90 ? -6 : 6), H - B - 8);
  }
}

function drawSeries(g, rows, s, X, Y, t0, t1, field) {
  const fps = S.meta.fps;
  const pass = (faintPass) => {
    g.strokeStyle = s.c;
    g.fillStyle = s.c;
    g.globalAlpha = faintPass ? 0.35 : 1;
    g.lineWidth = faintPass ? 1 : 1.7;
    g.beginPath();
    let prev = null;
    for (const r of rows) {
      const t = r.frame / fps;
      if (t < t0 - 0.05 || t > t1 + 0.05) { prev = null; continue; }
      let v = r[s.k];
      const skip = v === null || v === undefined || (s.hide && s.hide(r)) || (!faintPass && s.faint && s.faint(r));
      if (skip) { prev = null; continue; }
      const x = X(t), y = Y(v);
      if (s.dots) { g.moveTo(x + 1.6, y); g.arc(x, y, 1.6, 0, 2 * Math.PI); prev = r; continue; }
      const broken = !prev || r.frame - prev.frame > 1 || (!field && prev.segment !== r.segment);
      if (broken) g.moveTo(x, y); else g.lineTo(x, y);
      prev = r;
    }
    if (s.dots) g.fill(); else g.stroke();
    g.globalAlpha = 1;
  };
  if (s.faint) pass(true);
  pass(false);
}

function drawCuts(g, top, bot, X, t0, t1) {
  g.strokeStyle = AXIS; g.lineWidth = 1; g.setLineDash([3, 3]);
  for (const s of S.obj.segments.slice(1)) {
    const x = X(s.f0 / S.meta.fps);
    if (x < CH.L || x > CH.W - CH.R) continue;
    g.beginPath(); g.moveTo(x, top); g.lineTo(x, bot); g.stroke();
  }
  g.setLineDash([]);
}

function shadeUncertain(g, rows, top, bot, X, t0, t1) {
  g.fillStyle = 'rgba(255,159,28,0.12)';
  const dt = 1 / S.meta.fps;
  for (const r of rows) {
    if (r.field_quality !== 'uncertain') continue;
    const t = r.frame / S.meta.fps;
    if (t < t0 || t > t1) continue;
    g.fillRect(X(t), top, Math.max(1, X(t + dt) - X(t)), bot - top);
  }
}

function drawEvents(g, p, top, bot, X, t0, t1) {
  const kinds = [['connect', PAL[6], 'v'], ['separate', PAL[2], '^'], ['enter view', PAL[0], '>'], ['leave view', PAL[1], '<']];
  const rowH = (bot - top) / kinds.length;
  g.textAlign = 'right';
  kinds.forEach(([k, c], i) => {
    const y = top + rowH * (i + 0.5);
    g.fillStyle = INK2; g.fillText(k, CH.L - 6, y);
    g.strokeStyle = GRID; g.beginPath(); g.moveTo(CH.L, y); g.lineTo(CH.W - CH.R, y); g.stroke();
  });
  for (const e of S.field.events) {
    if (!e.counted) continue;
    const i = kinds.findIndex((k) => k[0] === e.type);
    if (i < 0) continue;
    const t = e.frame / S.meta.fps;
    if (t < t0 || t > t1) continue;
    const x = X(t), y = top + rowH * (i + 0.5);
    const r = 3 + 5 * Math.min(1, Math.sqrt((e.size || 0) / 3000));
    g.fillStyle = kinds[i][1];
    g.beginPath();
    const d = kinds[i][2];
    if (d === 'v') { g.moveTo(x - r, y - r); g.lineTo(x + r, y - r); g.lineTo(x, y + r); }
    else if (d === '^') { g.moveTo(x - r, y + r); g.lineTo(x + r, y + r); g.lineTo(x, y - r); }
    else if (d === '>') { g.moveTo(x - r, y - r); g.lineTo(x + r, y); g.lineTo(x - r, y + r); }
    else { g.moveTo(x + r, y - r); g.lineTo(x - r, y); g.lineTo(x + r, y + r); }
    g.closePath(); g.fill();
  }
}

function chartFrame(ev) {
  if (!CH) return null;
  const r = $('chart').getBoundingClientRect();
  const x = ev.clientX - r.left;
  if (x < CH.L || x > CH.W - CH.R) return null;
  const t = CH.t0 + (x - CH.L) / (CH.W - CH.L - CH.R) * (CH.t1 - CH.t0);
  return Math.max(0, Math.min(S.meta.n_frames - 1, Math.round(t * S.meta.fps)));
}

// ---------------------------------------------------------------- playback and controls
let lastTs = null, acc = 0;
function loop(ts) {
  if (!S.playing) { lastTs = null; return; }
  if (lastTs !== null) {
    acc += (ts - lastTs) / 1000 * S.meta.fps * Number($('speed').value);
    const n = Math.floor(acc);
    if (n >= 1) {
      acc -= n;
      if (S.frame + n >= S.meta.n_frames) { S.frame = S.meta.n_frames - 1; togglePlay(false); }
      else S.frame += n;
      drawFrame(); drawChart();
    }
  }
  lastTs = ts;
  requestAnimationFrame(loop);
}

function togglePlay(on) {
  S.playing = on === undefined ? !S.playing : on;
  $('b-play').textContent = S.playing ? 'Pause' : 'Play';
  if (S.playing) { acc = 0; requestAnimationFrame(loop); }
}

function objBounds() {
  if (!S.obj || !S.obj.rows.length) return [0, S.meta.n_frames - 1];
  return [S.obj.rows[0].frame, S.obj.rows[S.obj.rows.length - 1].frame];
}

function stepObject(d) {
  const ids = ['field', ...S.meta.objects.map((o) => o.object_id)];
  const i = ids.indexOf(S.sel);
  const j = Math.max(0, Math.min(ids.length - 1, i + d));
  if (j !== i) select(ids[j], { seek: true });
}

function bindUi() {
  $('b-play').onclick = () => togglePlay();
  $('b-back').onclick = () => seek(S.frame - 1);
  $('b-fwd').onclick = () => seek(S.frame + 1);
  $('b-back10').onclick = () => seek(S.frame - 10);
  $('b-fwd10').onclick = () => seek(S.frame + 10);
  $('b-start').onclick = () => seek(objBounds()[0]);
  $('b-end').onclick = () => seek(objBounds()[1]);
  for (const id of ['t-zoom', 't-path', 't-all']) $(id).onchange = () => drawFrame();
  const setRange = (r) => {
    S.range = r;
    $('r-obj').setAttribute('aria-pressed', String(r === 'obj'));
    $('r-all').setAttribute('aria-pressed', String(r === 'all'));
    drawChart();
  };
  $('r-obj').onclick = () => setRange('obj');
  $('r-all').onclick = () => setRange('all');
  const cv = $('chart');
  cv.addEventListener('mousemove', (e) => { S.hoverF = chartFrame(e); drawChart(); });
  cv.addEventListener('mouseleave', () => { S.hoverF = null; drawChart(); });
  cv.addEventListener('click', (e) => { const f = chartFrame(e); if (f !== null) { togglePlay(false); seek(f); } });
  window.addEventListener('resize', resizeChart);
  $('btn-notes').onclick = () => {
    const notes = S.meta.notes || [];
    $('dlg-body').innerHTML = '<ul>' + notes.map((n) => `<li>${n.replace(/&/g, '&amp;').replace(/</g, '&lt;')}</li>`).join('') + '</ul>'
      + `<p class="muted small">Self-checks: ${S.meta.all_checks_passed ? 'all passed' : 'SOME FAILED, see report.json'}.</p>`;
    $('dialog').hidden = false;
  };
  $('dlg-close').onclick = () => { $('dialog').hidden = true; };
  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, select, textarea')) return;
    if (!$('dialog').hidden) { if (e.key === 'Escape') $('dialog').hidden = true; return; }
    const k = e.key;
    if (k === ' ') { e.preventDefault(); togglePlay(); }
    else if (k === 'ArrowLeft') { e.preventDefault(); seek(S.frame - (e.shiftKey ? 10 : 1)); }
    else if (k === 'ArrowRight') { e.preventDefault(); seek(S.frame + (e.shiftKey ? 10 : 1)); }
    else if (k === 'Home') { e.preventDefault(); seek(objBounds()[0]); }
    else if (k === 'End') { e.preventDefault(); seek(objBounds()[1]); }
    else if (k === '[') stepObject(-1);
    else if (k === ']') stepObject(1);
    else if (k === 'z' || k === 'Z') { $('t-zoom').checked = !$('t-zoom').checked; drawFrame(); }
  });
}

init().catch((e) => { $('src').textContent = 'Error: ' + e.message; console.error(e); });
