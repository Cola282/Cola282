// Render frames of index.html with headless Chromium.
// node render.js <outDir> <w> <h> <fps> <dur> [workers] [sub] [frameList]
// frameList (optional): comma list of frame numbers (for stills / contact sheets)
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
const http = require('http');
const MIME = { '.html':'text/html', '.js':'text/javascript', '.json':'application/json', '.png':'image/png', '.otf':'font/otf', '.ttf':'font/ttf' };
function serve(root){ return new Promise(r => { const s = http.createServer((q, res) => { const f = path.join(root, decodeURIComponent(q.url.split('?')[0])); if (!f.startsWith(root) || !fs.existsSync(f)) { res.writeHead(404); return res.end(); } res.writeHead(200, { 'Content-Type': MIME[path.extname(f)] || 'application/octet-stream' }); fs.createReadStream(f).pipe(res); }); s.listen(0, '127.0.0.1', () => r(s)); }); }

(async () => {
  const [outDir, w, h, fps, dur, workers = 3, sub = 6, list] = process.argv.slice(2);
  fs.mkdirSync(outDir, { recursive: true });
  const total = Math.round(+dur * +fps);
  const frames = list ? list.split(',').map(Number) : [...Array(total).keys()];
  const server = await serve(path.join(__dirname, '..'));
  const browser = await chromium.launch();
  const url = `http://127.0.0.1:${server.address().port}/render/index.html?w=${w}&h=${h}&sub=${sub}`;
  let next = 0, done = 0; const t0 = Date.now();
  async function worker() {
    const page = await browser.newPage({ viewport: { width: 400, height: 300 } });
    page.on('pageerror', e => { console.error('PAGE ERROR', e.message); process.exit(1); });
    await page.goto(url);
    await page.waitForFunction('window.READY === true', null, { timeout: 60000 });
    while (next < frames.length) {
      const f = frames[next++];
      await page.evaluate(n => window.renderFrame(n), f);
      const data = await page.evaluate(() => window.frameData());
      fs.writeFileSync(path.join(outDir, `f_${String(f).padStart(5, '0')}.png`), Buffer.from(data.split(',')[1], 'base64'));
      done++;
      if (done % 60 === 0) console.log(`${done}/${frames.length}  ${((Date.now() - t0) / done).toFixed(0)} ms/frame`);
    }
    await page.close();
  }
  await Promise.all([...Array(+workers)].map(worker));
  await browser.close(); server.close();
  console.log('done', done, 'frames in', ((Date.now() - t0) / 1000).toFixed(1), 's');
})();
