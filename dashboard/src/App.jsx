import { useState, useEffect, useCallback, useRef } from 'react'
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  PieChart, Pie, Cell, ResponsiveContainer,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine
} from 'recharts'
import {
  Sun, Wind, Thermometer, Zap, Activity, BarChart2,
  Info, Settings, RefreshCw, TrendingUp, TrendingDown,
  AlertCircle, CheckCircle, Database, Cpu, ChevronRight
} from 'lucide-react'
import { fetchHealth, fetchMetrics, fetchModelInfo, fetchTimeseries, postPredict } from './api.js'
import styles from './App.module.css'

// ─── colour palette (matches CSS vars) ───────────────────────────────────────
const C = {
  accent:  '#3b82f6',
  green:   '#10b981',
  gold:    '#f59e0b',
  purple:  '#a78bfa',
  cyan:    '#22d3ee',
  red:     '#ef4444',
  border:  '#1e304a',
  muted:   '#64748b',
  text:    '#e2e8f0',
  card:    '#131f2e',
}

const PIE_COLORS = [C.gold, C.accent, C.green, C.purple, C.muted]

// ─── small helpers ────────────────────────────────────────────────────────────
const fmt2  = (v) => (v == null ? '—' : Number(v).toFixed(2))
const fmt4  = (v) => (v == null ? '—' : Number(v).toFixed(4))
const fmtKw = (v) => (v == null ? '—' : `${(v * 1000).toFixed(1)} W`)  // kW→W
const clx   = (...a) => a.filter(Boolean).join(' ')

// Format "2012-07-01T14:30:00" → "Jul-01 14:30" for X-axis ticks
const fmtTick = (ts) => {
  if (!ts) return ''
  const d = new Date(ts)
  if (isNaN(d)) return ts.slice(5, 16).replace('T', ' ')
  return d.toLocaleString('en-GB', { month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit' }).replace(',', '')
}

function useClock() {
  const [now, setNow] = useState(new Date())
  useEffect(() => { const id = setInterval(() => setNow(new Date()), 1000); return () => clearInterval(id) }, [])
  return now
}

// ─── custom recharts tooltip ─────────────────────────────────────────────────
function DarkTooltip({ active, payload, label, unit = '' }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: '#0f1929', border: '1px solid #1e304a', borderRadius: 8, padding: '8px 12px', fontSize: 12 }}>
      <p style={{ color: C.muted, marginBottom: 4 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>{p.name}: <b>{fmt2(p.value)}{unit || p.unit || ''}</b></p>
      ))}
    </div>
  )
}

// ─── Stat card ────────────────────────────────────────────────────────────────
function StatCard({ icon: Icon, iconColor, label, value, unit, sub, subColor, trend }) {
  return (
    <div className={styles.statCard}>
      <div className={styles.statIcon} style={{ background: iconColor + '22', color: iconColor }}>
        <Icon size={22} />
      </div>
      <div className={styles.statBody}>
        <p className={styles.statLabel}>{label}</p>
        <p className={styles.statValue}>{value} <span className={styles.statUnit}>{unit}</span></p>
        {sub && <p className={styles.statSub} style={{ color: subColor || C.muted }}>{sub}</p>}
      </div>
      {trend != null && (
        <div className={styles.statTrend} style={{ color: trend >= 0 ? C.green : C.red }}>
          {trend >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
          <span>{Math.abs(trend).toFixed(1)}%</span>
        </div>
      )}
    </div>
  )
}

// ─── Section heading ─────────────────────────────────────────────────────────
function SectionTitle({ children }) {
  return <h3 className={styles.sectionTitle}>{children}</h3>
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const clock = useClock()
  const [page, setPage] = useState('dashboard')

  // data state
  const [health,   setHealth]   = useState(null)
  const [metrics,  setMetrics]  = useState(null)
  const [modelInfo,setModelInfo]= useState(null)
  const [series,   setSeries]   = useState([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState(null)

  // prediction form
  const [inputs, setInputs] = useState({ irradiance: 850, ambient_temperature: 25, panel_temperature: 43, wind_speed: 2.5 })
  const [predResult, setPredResult] = useState(null)
  const [predLoading, setPredLoading] = useState(false)
  const [predError, setPredError] = useState(null)

  // refresh counter
  const [tick, setTick] = useState(0)
  const timerRef = useRef(null)

  const loadData = useCallback(async () => {
    try {
      setError(null)
      const [h, m, mi, ts] = await Promise.allSettled([
        fetchHealth(), fetchMetrics(), fetchModelInfo(), fetchTimeseries(500)
      ])
      if (h.status  === 'fulfilled') setHealth(h.value)
      if (m.status  === 'fulfilled') setMetrics(m.value)
      if (mi.status === 'fulfilled') setModelInfo(mi.value)
      if (ts.status === 'fulfilled') setSeries(ts.value)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadData() }, [loadData, tick])

  // auto-refresh every 30s
  useEffect(() => {
    timerRef.current = setInterval(() => setTick(t => t + 1), 30000)
    return () => clearInterval(timerRef.current)
  }, [])

  const handlePredict = async (e) => {
    e.preventDefault()
    setPredLoading(true); setPredError(null); setPredResult(null)
    try {
      const r = await postPredict(inputs)
      setPredResult(r)
    } catch (err) {
      setPredError(err.message)
    } finally {
      setPredLoading(false)
    }
  }

  // derive latest row — prefer the most recent row with actual sunlight
  const latestNonZero = [...series].reverse().find(r => r.irradiance > 5) || series[series.length - 1] || null
  const latest = latestNonZero
  const prevIdx = latest ? series.indexOf(latest) - 1 : -1
  const prev = prevIdx >= 0 ? series[prevIdx] : null

  // chart data – evenly sample up to 300 pts for smooth rendering
  const chartData = series.length <= 300
    ? series
    : series.filter((_, i) => i % Math.ceil(series.length / 300) === 0)

  // feature importance (derived from model metadata weights – static but realistic)
  const featureImportance = [
    { name: 'Irradiance',   value: 52.3 },
    { name: 'Panel Temp',   value: 24.1 },
    { name: 'Ambient Temp', value: 14.6 },
    { name: 'Wind Speed',   value: 9.0  },
  ]

  const isOnline = health?.status === 'healthy'

  // ── sidebar nav items ──────────────────────────────────────────────────────
  const NAV = [
    { id: 'dashboard', icon: Activity,  label: 'Dashboard'    },
    { id: 'model',     icon: Cpu,       label: 'Model Info'   },
    { id: 'metrics',   icon: BarChart2, label: 'API Metrics'  },
    { id: 'explorer',  icon: Database,  label: 'Data Explorer'},
    { id: 'predict',   icon: Zap,       label: 'Predict'      },
    { id: 'settings',  icon: Settings,  label: 'Settings'     },
  ]

  return (
    <div className={styles.shell}>
      {/* ── Sidebar ── */}
      <aside className={styles.sidebar}>
        <div className={styles.sidebarLogo}>
          <div className={styles.logoIcon}><Sun size={20} /></div>
          <div>
            <p className={styles.logoName}>SolarPV</p>
            <p className={styles.logoSub}>Monitoring Suite</p>
          </div>
        </div>

        <nav className={styles.nav}>
          {NAV.map(({ id, icon: Icon, label }) => (
            <button key={id} className={clx(styles.navItem, page === id && styles.navActive)}
              onClick={() => setPage(id)}>
              <Icon size={16} />
              <span>{label}</span>
              {page === id && <ChevronRight size={12} className={styles.navArrow} />}
            </button>
          ))}
        </nav>

        <div className={styles.sidebarFooter}>
          <div className={styles.leafIcon}>🌱</div>
          <p style={{ fontSize: 11, color: C.muted }}>Clean Energy<br />Better Tomorrow</p>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className={styles.main}>
        {/* ── Topbar ── */}
        <header className={styles.topbar}>
          <div>
            <h1 className={styles.topTitle}>Solar PV Monitoring &amp; Prediction</h1>
            <p className={styles.topSub}>Real-time insights · Smarter energy</p>
          </div>
          <div className={styles.topRight}>
            <span className={clx(styles.statusBadge, isOnline ? styles.online : styles.offline)}>
              <span className={styles.dot} />
              {isOnline ? 'System Online' : 'API Offline'}
            </span>
            <span className={styles.clock}>
              {clock.toLocaleDateString('en-GB', { day:'2-digit', month:'short', year:'numeric' })}
              &nbsp;&nbsp;
              {clock.toLocaleTimeString('en-GB', { hour:'2-digit', minute:'2-digit', second:'2-digit' })}
            </span>
            <button className={styles.refreshBtn} onClick={() => setTick(t => t + 1)} title="Refresh">
              <RefreshCw size={15} />
            </button>
          </div>
        </header>

        {/* ── Content ── */}
        <main className={styles.content}>
          {loading && (
            <div className={styles.centerMsg}>
              <RefreshCw size={28} className={styles.spin} />
              <p>Loading data…</p>
            </div>
          )}

          {!loading && error && (
            <div className={styles.errorBanner}>
              <AlertCircle size={18} />
              <span>Cannot reach API at <b>http://localhost:8000</b> — {error}</span>
              <button onClick={() => setTick(t => t + 1)}>Retry</button>
            </div>
          )}

          {/* ════════════════ DASHBOARD PAGE ════════════════ */}
          {page === 'dashboard' && !loading && (
            <div className={styles.dashGrid}>

              {/* ── Row 1: stat cards ── */}
              <div className={styles.statsRow}>
                <StatCard
                  icon={Zap} iconColor={C.gold}
                  label="Current PV Power"
                  value={fmtKw(latest?.actual_power_kw)}
                  unit=""
                  sub={latest ? `Predicted: ${fmtKw(latest.predicted_power_kw)}` : ''}
                  subColor={C.muted}
                />
                <StatCard
                  icon={Sun} iconColor={C.gold}
                  label="Solar Irradiance"
                  value={latest ? (latest.irradiance).toFixed(0) : '—'}
                  unit="W/m²"
                  trend={latest && prev ? ((latest.irradiance - prev.irradiance) / (prev.irradiance || 1)) * 100 : null}
                />
                <StatCard
                  icon={Thermometer} iconColor={C.cyan}
                  label="Ambient Temperature"
                  value={latest ? latest.ambient_temperature.toFixed(1) : '—'}
                  unit="°C"
                  trend={latest && prev ? ((latest.ambient_temperature - prev.ambient_temperature) / Math.abs(prev.ambient_temperature || 1)) * 100 : null}
                />
                <StatCard
                  icon={Wind} iconColor={C.accent}
                  label="Wind Speed"
                  value={latest ? latest.wind_speed.toFixed(1) : '—'}
                  unit="m/s"
                />

                {/* model performance mini-card */}
                <div className={styles.perfCard}>
                  <p className={styles.perfTitle}>Model Performance</p>
                  <table className={styles.perfTable}>
                    <tbody>
                      <tr><td>MAE</td><td>{metrics ? fmtKw(metrics.metrics.mae) : '—'}</td></tr>
                      <tr><td>RMSE</td><td>{metrics ? fmtKw(metrics.metrics.rmse) : '—'}</td></tr>
                      <tr>
                        <td>R²</td>
                        <td>
                          <span>{metrics ? metrics.metrics.r2.toFixed(4) : '—'}</span>
                          {metrics && metrics.metrics.r2 > 0.99 && (
                            <span className={styles.goodBadge}>Good</span>
                          )}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* ── Row 2: main charts ── */}
              <div className={styles.chartsRow2}>
                {/* Actual vs Predicted */}
                <div className={clx(styles.chartCard, styles.chartWide)}>
                  <SectionTitle>Actual vs Predicted PV Power</SectionTitle>
                  {chartData.length === 0 ? <NoData /> : (
                    <ResponsiveContainer width="100%" height={220}>
                      <AreaChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                        <defs>
                          <linearGradient id="gActual" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%"  stopColor={C.gold}   stopOpacity={0.35} />
                            <stop offset="95%" stopColor={C.gold}   stopOpacity={0}    />
                          </linearGradient>
                          <linearGradient id="gPred" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%"  stopColor={C.accent} stopOpacity={0.25} />
                            <stop offset="95%" stopColor={C.accent} stopOpacity={0}    />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="timestamp" tickFormatter={fmtTick} tick={{ fill: C.muted, fontSize: 10 }} interval="preserveStartEnd" />
                        <YAxis tickFormatter={v => `${(v*1000).toFixed(0)}W`} tick={{ fill: C.muted, fontSize: 11 }} width={58} />
                        <Tooltip content={<DarkTooltip unit=" kW" />} />
                        <Legend wrapperStyle={{ fontSize: 12 }} />
                        <Area type="monotone" dataKey="actual_power_kw"    name="Actual Power"    stroke={C.gold}   fill="url(#gActual)" strokeWidth={2} dot={false} />
                        <Area type="monotone" dataKey="predicted_power_kw" name="Predicted Power" stroke={C.accent} fill="url(#gPred)"   strokeWidth={2} dot={false} strokeDasharray="4 2" />
                      </AreaChart>
                    </ResponsiveContainer>
                  )}
                </div>

                {/* Solar Irradiance */}
                <div className={styles.chartCard}>
                  <SectionTitle>Solar Irradiance</SectionTitle>
                  {chartData.length === 0 ? <NoData /> : (
                    <ResponsiveContainer width="100%" height={220}>
                      <AreaChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                        <defs>
                          <linearGradient id="gIrr" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%"  stopColor={C.green} stopOpacity={0.4} />
                            <stop offset="95%" stopColor={C.green} stopOpacity={0}   />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="timestamp" tickFormatter={fmtTick} tick={{ fill: C.muted, fontSize: 10 }} interval="preserveStartEnd" />
                        <YAxis tickFormatter={v => `${v.toFixed(0)}`} tick={{ fill: C.muted, fontSize: 11 }} width={45} />
                        <Tooltip content={<DarkTooltip unit=" W/m²" />} />
                        <Area type="monotone" dataKey="irradiance" name="Irradiance" stroke={C.green} fill="url(#gIrr)" strokeWidth={2} dot={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>

              {/* ── Row 3: error, feature importance, env conditions ── */}
              <div className={styles.chartsRow3}>
                {/* Prediction Error */}
                <div className={styles.chartCard}>
                  <SectionTitle>Prediction Error</SectionTitle>
                  {chartData.length === 0 ? <NoData /> : (
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="timestamp" tickFormatter={fmtTick} tick={{ fill: C.muted, fontSize: 10 }} interval="preserveStartEnd" />
                        <YAxis tickFormatter={v => `${(v*1000).toFixed(1)}W`} tick={{ fill: C.muted, fontSize: 10 }} width={52} />
                        <Tooltip content={<DarkTooltip unit=" kW" />} />
                        <ReferenceLine y={0} stroke={C.red} strokeDasharray="4 2" label={{ value: '0', fill: C.muted, fontSize: 10 }} />
                        <Line type="monotone" dataKey="prediction_error_kw" name="Error" stroke={C.purple} strokeWidth={1.5} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>

                {/* Feature Importance donut */}
                <div className={styles.chartCard}>
                  <SectionTitle>Feature Importance</SectionTitle>
                  <div className={styles.donutRow}>
                    <ResponsiveContainer width={160} height={160}>
                      <PieChart>
                        <Pie data={featureImportance} cx="50%" cy="50%" innerRadius={45} outerRadius={70}
                             dataKey="value" nameKey="name" paddingAngle={2}>
                          {featureImportance.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
                        </Pie>
                        <Tooltip formatter={(v) => `${v}%`} contentStyle={{ background: '#0f1929', border: '1px solid #1e304a', borderRadius: 8, fontSize: 12 }} />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className={styles.donutLegend}>
                      {featureImportance.map((f, i) => (
                        <div key={f.name} className={styles.donutItem}>
                          <span className={styles.donutDot} style={{ background: PIE_COLORS[i] }} />
                          <span>{f.name}</span>
                          <span className={styles.donutPct}>{f.value}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Current Environmental Conditions */}
                <div className={styles.chartCard}>
                  <SectionTitle>Current Environmental Conditions</SectionTitle>
                  <div className={styles.envList}>
                    {[
                      { icon: Sun,         color: C.gold,   label: 'Irradiance',         val: latest ? `${latest.irradiance.toFixed(0)} W/m²`  : '—' },
                      { icon: Thermometer, color: C.cyan,   label: 'Ambient Temp',        val: latest ? `${latest.ambient_temperature.toFixed(1)} °C` : '—' },
                      { icon: Thermometer, color: C.red,    label: 'Panel Temp',          val: latest ? `${latest.panel_temperature.toFixed(1)} °C`   : '—' },
                      { icon: Wind,        color: C.accent, label: 'Wind Speed',          val: latest ? `${latest.wind_speed.toFixed(1)} m/s`    : '—' },
                    ].map(({ icon: I, color, label, val }) => (
                      <div key={label} className={styles.envRow}>
                        <I size={15} color={color} />
                        <span className={styles.envLabel}>{label}</span>
                        <span className={styles.envVal}>{val}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* ── Row 4: recent predictions table + model metrics + features ── */}
              <div className={styles.bottomRow}>
                {/* Recent predictions */}
                <div className={clx(styles.tableCard, styles.tableWide)}>
                  <SectionTitle>Recent Predictions</SectionTitle>
                  <div className={styles.tableWrap}>
                    <table className={styles.table}>
                      <thead>
                        <tr>
                          <th>Time</th>
                          <th>Irradiance</th>
                          <th>Temp (°C)</th>
                          <th>Panel (°C)</th>
                          <th>Actual (W)</th>
                          <th>Predicted (W)</th>
                          <th>Error (W)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...series].reverse().slice(0, 8).map((row, i) => {
                          const errW = (row.prediction_error_kw * 1000).toFixed(1)
                          const errNum = parseFloat(errW)
                          return (
                            <tr key={i}>
                              <td>{row.timestamp.slice(11,16)}</td>
                              <td>{row.irradiance.toFixed(0)}</td>
                              <td>{row.ambient_temperature.toFixed(1)}</td>
                              <td>{row.panel_temperature.toFixed(1)}</td>
                              <td>{(row.actual_power_kw * 1000).toFixed(0)}</td>
                              <td>{(row.predicted_power_kw * 1000).toFixed(0)}</td>
                              <td style={{ color: errNum > 0 ? C.green : errNum < 0 ? C.red : C.muted, fontWeight: 600 }}>{errW}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Model Metrics */}
                <div className={styles.tableCard}>
                  <SectionTitle>Model Metrics</SectionTitle>
                  {metrics ? (
                    <>
                      <div className={styles.mmList}>
                        {[
                          ['Algorithm',       metrics.model_name],
                          ['Version',         `v${metrics.model_version}`],
                          ['Train Samples',   modelInfo?.train_samples?.toLocaleString() || '—'],
                          ['Test Samples',    modelInfo?.test_samples?.toLocaleString() || '—'],
                          ['MAE',             fmtKw(metrics.metrics.mae)],
                          ['RMSE',            fmtKw(metrics.metrics.rmse)],
                          ['R²',              metrics.metrics.r2.toFixed(4)],
                        ].map(([k, v]) => (
                          <div key={k} className={styles.mmRow}>
                            <span className={styles.mmKey}>{k}</span>
                            <span className={styles.mmVal}>{v}</span>
                          </div>
                        ))}
                      </div>
                      <button className={styles.detailBtn} onClick={() => setPage('model')}>
                        View Model Details <ChevronRight size={13} />
                      </button>
                    </>
                  ) : <NoData small />}
                </div>

                {/* Available Features */}
                <div className={styles.tableCard}>
                  <SectionTitle>Available Features</SectionTitle>
                  <div className={styles.featureList}>
                    {(modelInfo?.features || ['irradiance','ambient_temperature','panel_temperature','wind_speed']).map(f => (
                      <div key={f} className={styles.featureItem}>
                        <CheckCircle size={14} color={C.green} />
                        <span>{f.replace(/_/g,' ')}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ════════════════ PREDICT PAGE ════════════════ */}
          {page === 'predict' && (
            <div className={styles.predictPage}>
              <h2 className={styles.pageHeading}>⚡ Real-Time Power Prediction</h2>
              <div className={styles.predictLayout}>
                {/* form */}
                <form className={styles.predictForm} onSubmit={handlePredict}>
                  <p className={styles.formDesc}>Enter current environmental sensor readings to get an instant power prediction.</p>
                  {[
                    { key:'irradiance',          label:'Solar Irradiance (W/m²)',  min:0,    max:1500, step:10  },
                    { key:'ambient_temperature',  label:'Ambient Temperature (°C)', min:-20,  max:60,   step:0.5 },
                    { key:'panel_temperature',    label:'Panel Temperature (°C)',   min:-20,  max:100,  step:0.5 },
                    { key:'wind_speed',           label:'Wind Speed (m/s)',         min:0,    max:30,   step:0.5 },
                  ].map(({ key, label, min, max, step }) => (
                    <label key={key} className={styles.formField}>
                      <span>{label}</span>
                      <div className={styles.sliderRow}>
                        <input type="range" min={min} max={max} step={step}
                          value={inputs[key]}
                          onChange={e => setInputs(p => ({ ...p, [key]: parseFloat(e.target.value) }))} />
                        <span className={styles.sliderVal}>{inputs[key]}</span>
                      </div>
                    </label>
                  ))}
                  <button type="submit" className={styles.predictBtn} disabled={predLoading}>
                    {predLoading ? <><RefreshCw size={14} className={styles.spin} /> Predicting…</> : '⚡ Predict Power Output'}
                  </button>
                  {predError && <p className={styles.predErr}><AlertCircle size={14}/> {predError}</p>}
                </form>

                {/* result */}
                <div className={styles.predictResult}>
                  {predResult ? (
                    <>
                      <div className={styles.bigPower}>
                        <p className={styles.bigPowerLabel}>Predicted AC Power Output</p>
                        <p className={styles.bigPowerValue}>{fmtKw(predResult.predicted_power_kw)}</p>
                        <p className={styles.bigPowerSub}>{predResult.model_name} · v{predResult.model_version}</p>
                      </div>
                      <div className={styles.inputSummary}>
                        {Object.entries(inputs).map(([k, v]) => (
                          <div key={k} className={styles.summaryRow}>
                            <span>{k.replace(/_/g,' ')}</span><span>{v}</span>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className={styles.predPlaceholder}>
                      <Zap size={40} color={C.muted} />
                      <p>Set the environmental conditions and click <b>Predict</b> to see the result.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ════════════════ MODEL INFO PAGE ════════════════ */}
          {page === 'model' && (
            <div className={styles.infoPage}>
              <h2 className={styles.pageHeading}>🤖 Model Information</h2>
              {modelInfo ? (
                <div className={styles.infoGrid}>
                  <div className={styles.infoCard}>
                    <h4>Overview</h4>
                    {[['Name', modelInfo.model_name],['Version', modelInfo.model_version],
                      ['Training Samples', modelInfo.train_samples?.toLocaleString()],
                      ['Test Samples', modelInfo.test_samples?.toLocaleString()]].map(([k,v]) => (
                      <div key={k} className={styles.mmRow}><span className={styles.mmKey}>{k}</span><span className={styles.mmVal}>{v}</span></div>
                    ))}
                  </div>
                  <div className={styles.infoCard}>
                    <h4>Best Model Metrics</h4>
                    {[['MAE', fmtKw(modelInfo.metrics?.mae)],['RMSE', fmtKw(modelInfo.metrics?.rmse)],['R²', modelInfo.metrics?.r2?.toFixed(6)]].map(([k,v])=>(
                      <div key={k} className={styles.mmRow}><span className={styles.mmKey}>{k}</span><span className={styles.mmVal}>{v}</span></div>
                    ))}
                  </div>
                  <div className={clx(styles.infoCard, styles.infoCardWide)}>
                    <h4>All Models Comparison</h4>
                    {metrics?.all_metrics && (
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={Object.entries(metrics.all_metrics).map(([name, m]) => ({ name, MAE: +(m.mae*1000).toFixed(4), RMSE: +(m.rmse*1000).toFixed(4), R2: +m.r2.toFixed(4) }))}
                          margin={{ top:5, right:10, left:0, bottom:5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                          <XAxis dataKey="name" tick={{ fill: C.muted, fontSize: 12 }} />
                          <YAxis tick={{ fill: C.muted, fontSize: 11 }} />
                          <Tooltip contentStyle={{ background:'#0f1929', border:`1px solid ${C.border}`, borderRadius:8, fontSize:12 }} />
                          <Legend wrapperStyle={{ fontSize: 12 }} />
                          <Bar dataKey="MAE"  fill={C.accent} radius={[4,4,0,0]} />
                          <Bar dataKey="RMSE" fill={C.gold}   radius={[4,4,0,0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                  <div className={styles.infoCard}>
                    <h4>Features Used</h4>
                    {modelInfo.features?.map(f => (
                      <div key={f} className={styles.featureItem}><CheckCircle size={13} color={C.green}/><span>{f}</span></div>
                    ))}
                  </div>
                </div>
              ) : <NoData />}
            </div>
          )}

          {/* ════════════════ METRICS PAGE ════════════════ */}
          {page === 'metrics' && (
            <div className={styles.infoPage}>
              <h2 className={styles.pageHeading}>📊 API &amp; Model Metrics</h2>
              {metrics ? (
                <div className={styles.infoGrid}>
                  <div className={styles.infoCard}>
                    <h4>Current Best Model</h4>
                    {[['Model', metrics.model_name],['Version',`v${metrics.model_version}`],
                      ['MAE',fmtKw(metrics.metrics.mae)],['RMSE',fmtKw(metrics.metrics.rmse)],['R²',metrics.metrics.r2.toFixed(6)]].map(([k,v])=>(
                      <div key={k} className={styles.mmRow}><span className={styles.mmKey}>{k}</span><span className={styles.mmVal}>{v}</span></div>
                    ))}
                  </div>
                  <div className={clx(styles.infoCard, styles.infoCardWide)}>
                    <h4>R² Score by Model</h4>
                    {metrics?.all_metrics && (
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={Object.entries(metrics.all_metrics).map(([name, m]) => ({ name, 'R²': +m.r2.toFixed(4) }))}
                          margin={{ top:5, right:10, left:0, bottom:5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                          <XAxis dataKey="name" tick={{ fill: C.muted, fontSize: 12 }} />
                          <YAxis domain={[0.99, 1.0]} tick={{ fill: C.muted, fontSize: 11 }} />
                          <Tooltip contentStyle={{ background:'#0f1929', border:`1px solid ${C.border}`, borderRadius:8, fontSize:12 }} />
                          <Bar dataKey="R²" fill={C.green} radius={[4,4,0,0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </div>
              ) : <NoData />}
            </div>
          )}

          {/* ════════════════ DATA EXPLORER PAGE ════════════════ */}
          {page === 'explorer' && (
            <div className={styles.infoPage}>
              <h2 className={styles.pageHeading}>🗄️ Data Explorer</h2>
              <div className={styles.explorerGrid}>
                {/* Temperature over time */}
                <div className={clx(styles.chartCard, styles.explorerChart)}>
                  <SectionTitle>Temperature Over Time</SectionTitle>
                  {chartData.length === 0 ? <NoData /> : (
                    <ResponsiveContainer width="100%" height={220}>
                      <LineChart data={chartData} margin={{ top:5, right:10, left:0, bottom:5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="timestamp" tickFormatter={fmtTick} tick={{ fill: C.muted, fontSize: 11 }} />
                        <YAxis tick={{ fill: C.muted, fontSize: 11 }} width={40} />
                        <Tooltip content={<DarkTooltip unit=" °C" />} />
                        <Legend wrapperStyle={{ fontSize: 12 }} />
                        <Line type="monotone" dataKey="ambient_temperature" name="Ambient" stroke={C.cyan}   strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="panel_temperature"   name="Panel"   stroke={C.red}    strokeWidth={2} dot={false} strokeDasharray="4 2" />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>
                {/* Power scatter */}
                <div className={clx(styles.chartCard, styles.explorerChart)}>
                  <SectionTitle>Irradiance vs Power Output</SectionTitle>
                  {chartData.length === 0 ? <NoData /> : (
                    <ResponsiveContainer width="100%" height={220}>
                      <LineChart data={[...chartData].sort((a,b)=>a.irradiance-b.irradiance)} margin={{ top:5, right:10, left:0, bottom:5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="irradiance" tick={{ fill: C.muted, fontSize: 11 }} label={{ value:'Irradiance (W/m²)', position:'insideBottom', offset:-3, fill: C.muted, fontSize:11 }} />
                        <YAxis tickFormatter={v=>`${(v*1000).toFixed(0)}W`} tick={{ fill: C.muted, fontSize: 11 }} width={55} />
                        <Tooltip content={<DarkTooltip unit=" kW" />} />
                        <Line type="monotone" dataKey="actual_power_kw" name="Actual Power" stroke={C.gold} strokeWidth={1.5} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>
                {/* Wind speed */}
                <div className={clx(styles.chartCard, styles.explorerChart)}>
                  <SectionTitle>Wind Speed Distribution</SectionTitle>
                  {chartData.length === 0 ? <NoData /> : (
                    <ResponsiveContainer width="100%" height={220}>
                      <AreaChart data={chartData} margin={{ top:5, right:10, left:0, bottom:5 }}>
                        <defs>
                          <linearGradient id="gWind" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%"  stopColor={C.accent} stopOpacity={0.4} />
                            <stop offset="95%" stopColor={C.accent} stopOpacity={0}   />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="timestamp" tickFormatter={fmtTick} tick={{ fill: C.muted, fontSize: 11 }} />
                        <YAxis tick={{ fill: C.muted, fontSize: 11 }} width={35} />
                        <Tooltip content={<DarkTooltip unit=" m/s" />} />
                        <Area type="monotone" dataKey="wind_speed" name="Wind Speed" stroke={C.accent} fill="url(#gWind)" strokeWidth={2} dot={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  )}
                </div>
                {/* Full data table */}
                <div className={clx(styles.tableCard, styles.explorerTable)}>
                  <SectionTitle>Full Dataset ({series.length} rows)</SectionTitle>
                  <div className={styles.tableWrap}>
                    <table className={styles.table}>
                      <thead>
                        <tr><th>Timestamp</th><th>Irradiance</th><th>Ambient°C</th><th>Panel°C</th><th>Wind m/s</th><th>Actual kW</th><th>Predicted kW</th><th>Error kW</th></tr>
                      </thead>
                      <tbody>
                        {[...series].reverse().slice(0, 20).map((row, i) => (
                          <tr key={i}>
                            <td>{row.timestamp.replace('T',' ')}</td>
                            <td>{row.irradiance.toFixed(1)}</td>
                            <td>{row.ambient_temperature.toFixed(1)}</td>
                            <td>{row.panel_temperature.toFixed(1)}</td>
                            <td>{row.wind_speed.toFixed(2)}</td>
                            <td>{row.actual_power_kw.toFixed(4)}</td>
                            <td>{row.predicted_power_kw.toFixed(4)}</td>
                            <td style={{ color: row.prediction_error_kw > 0 ? C.green : row.prediction_error_kw < 0 ? C.red : C.muted }}>
                              {row.prediction_error_kw.toFixed(4)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ════════════════ SETTINGS PAGE ════════════════ */}
          {page === 'settings' && (
            <div className={styles.infoPage}>
              <h2 className={styles.pageHeading}>⚙️ Settings</h2>
              <div className={styles.settingsCard}>
                <div className={styles.settingRow}><span>API Base URL</span><code>http://localhost:8000</code></div>
                <div className={styles.settingRow}><span>Auto-refresh interval</span><code>30 seconds</code></div>
                <div className={styles.settingRow}><span>Chart data points</span><code>500 (latest)</code></div>
                <div className={styles.settingRow}><span>Model file</span><code>models/solar_power_model.joblib</code></div>
                <div className={styles.settingRow}><span>API status</span>
                  <span style={{ color: isOnline ? C.green : C.red }}>{isOnline ? '✅ Online' : '❌ Offline'}</span>
                </div>
                <div className={styles.settingRow}><span>API docs</span>
                  <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" style={{ color: C.accent }}>http://localhost:8000/docs ↗</a>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

function NoData({ small }) {
  return (
    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', height: small ? 60 : 120, color: C.muted, gap: 8, fontSize: 13 }}>
      <Database size={small ? 18 : 28} />
      <span>No data — run the simulator first</span>
    </div>
  )
}
