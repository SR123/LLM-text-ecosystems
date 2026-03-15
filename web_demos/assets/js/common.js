export function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

export function drawSeries(canvas, series, color = '#a33928') {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = '#ddd';
  ctx.lineWidth = 1;
  for (let i = 0; i < 6; i++) {
    const y = (h - 30) * (i / 5) + 10;
    ctx.beginPath();
    ctx.moveTo(30, y);
    ctx.lineTo(w - 10, y);
    ctx.stroke();
  }

  const maxX = Math.max(1, series.length - 1);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  series.forEach((v, i) => {
    const x = 30 + (w - 40) * (i / maxX);
    const y = 10 + (h - 30) * (1 - v);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}
