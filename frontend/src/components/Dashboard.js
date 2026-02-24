import React, { useState, useEffect, useCallback } from 'react';
import {
  Box, Grid, Paper, Typography, Button, Slider, Select, MenuItem,
  FormControl, InputLabel, TextField, Collapse, Alert, CircularProgress,
  Card, CardContent,
} from '@mui/material';
import { ExpandMore, ExpandLess } from '@mui/icons-material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend, Filler,
} from 'chart.js';
import { generateForecast, getLatestForecast, getDataSummary } from '../services/api';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

function MetricsCard({ title, value, subtitle }) {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography color="text.secondary" gutterBottom variant="body2">{title}</Typography>
        <Typography variant="h4" component="div">{value}</Typography>
        {subtitle && <Typography variant="body2" color="text.secondary">{subtitle}</Typography>}
      </CardContent>
    </Card>
  );
}

export default function Dashboard() {
  const [forecastDays, setForecastDays] = useState(60);
  const [confidence, setConfidence] = useState(0.9);
  const [startDate, setStartDate] = useState('');
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [delay, setDelay] = useState(2);
  const [historyDays, setHistoryDays] = useState(365);
  const [ensembleSize, setEnsembleSize] = useState(50);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [forecast, setForecast] = useState(null);
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    getDataSummary({}).then(r => setSummary(r.data)).catch(() => {});
    getLatestForecast(60).then(r => setForecast(r.data)).catch(() => {});
  }, []);

  const handleGenerate = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await generateForecast({
        start_date: startDate || new Date().toISOString().split('T')[0],
        forecast_days: forecastDays,
        confidence_levels: [0.80, 0.90, 0.95],
        model_config: { delay, history_days: historyDays, ensemble_size: ensembleSize },
      });
      setForecast(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Forecast generation failed.');
    } finally {
      setLoading(false);
    }
  }, [startDate, forecastDays, delay, historyDays, ensembleSize]);

  // Build chart data
  const chartData = forecast?.predictions ? {
    labels: forecast.predictions.map(p => p.date),
    datasets: [
      {
        label: 'Predicted Occupancy',
        data: forecast.predictions.map(p => p.predicted),
        borderColor: '#1565c0',
        backgroundColor: 'rgba(21,101,192,0.1)',
        borderWidth: 2,
        pointRadius: 1,
        fill: false,
      },
      {
        label: '90% CI Upper',
        data: forecast.predictions.map(p => p.upper_90),
        borderColor: 'transparent',
        backgroundColor: 'rgba(21,101,192,0.1)',
        pointRadius: 0,
        fill: '+1',
      },
      {
        label: '90% CI Lower',
        data: forecast.predictions.map(p => p.lower_90),
        borderColor: 'transparent',
        backgroundColor: 'rgba(21,101,192,0.1)',
        pointRadius: 0,
        fill: false,
      },
      {
        label: '80% CI Upper',
        data: forecast.predictions.map(p => p.upper_80),
        borderColor: 'transparent',
        backgroundColor: 'rgba(21,101,192,0.15)',
        pointRadius: 0,
        fill: '+1',
      },
      {
        label: '80% CI Lower',
        data: forecast.predictions.map(p => p.lower_80),
        borderColor: 'transparent',
        backgroundColor: 'rgba(21,101,192,0.15)',
        pointRadius: 0,
        fill: false,
      },
    ],
  } : null;

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'Bed Occupancy Forecast' },
      tooltip: { mode: 'index', intersect: false },
    },
    scales: {
      x: { title: { display: true, text: 'Date' }, ticks: { maxTicksLimit: 15 } },
      y: { title: { display: true, text: 'Beds Occupied' }, beginAtZero: false },
    },
  };

  const preds = forecast?.predictions || [];
  const predValues = preds.map(p => p.predicted);
  const avgPred = predValues.length ? (predValues.reduce((a, b) => a + b, 0) / predValues.length).toFixed(0) : '-';
  const peakIdx = predValues.indexOf(Math.max(...predValues));
  const minIdx = predValues.indexOf(Math.min(...predValues));
  const metrics = forecast?.model_metrics || forecast?.metrics;

  return (
    <Box>
      <Typography variant="h5" gutterBottom>Bed Occupancy Forecasting</Typography>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Controls */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={3}>
            <TextField
              label="Start Date" type="date" fullWidth
              InputLabelProps={{ shrink: true }}
              value={startDate} onChange={e => setStartDate(e.target.value)}
            />
          </Grid>
          <Grid item xs={12} sm={3}>
            <Typography gutterBottom>Forecast Days: {forecastDays}</Typography>
            <Slider value={forecastDays} onChange={(_, v) => setForecastDays(v)} min={1} max={365} />
          </Grid>
          <Grid item xs={12} sm={3}>
            <FormControl fullWidth>
              <InputLabel>Confidence Level</InputLabel>
              <Select value={confidence} label="Confidence Level" onChange={e => setConfidence(e.target.value)}>
                <MenuItem value={0.80}>80%</MenuItem>
                <MenuItem value={0.90}>90%</MenuItem>
                <MenuItem value={0.95}>95%</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={3}>
            <Button
              variant="contained" fullWidth size="large"
              onClick={handleGenerate} disabled={loading}
              sx={{ height: 56 }}
            >
              {loading ? <CircularProgress size={24} color="inherit" /> : 'Generate Forecast'}
            </Button>
          </Grid>
        </Grid>

        <Button
          size="small" onClick={() => setAdvancedOpen(!advancedOpen)}
          endIcon={advancedOpen ? <ExpandLess /> : <ExpandMore />}
          sx={{ mt: 1 }}
        >
          Advanced Settings
        </Button>
        <Collapse in={advancedOpen}>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={4}>
              <Typography gutterBottom>Delay: {delay}</Typography>
              <Slider value={delay} onChange={(_, v) => setDelay(v)} min={1} max={7} />
            </Grid>
            <Grid item xs={4}>
              <Typography gutterBottom>History Days: {historyDays}</Typography>
              <Slider value={historyDays} onChange={(_, v) => setHistoryDays(v)} min={180} max={1825} step={30} />
            </Grid>
            <Grid item xs={4}>
              <Typography gutterBottom>Ensemble Size: {ensembleSize}</Typography>
              <Slider value={ensembleSize} onChange={(_, v) => setEnsembleSize(v)} min={10} max={100} />
            </Grid>
          </Grid>
        </Collapse>
      </Paper>

      {/* Chart */}
      {chartData && (
        <Paper sx={{ p: 2, mb: 3, height: 500 }}>
          <Line data={chartData} options={chartOptions} />
        </Paper>
      )}

      {/* Metrics cards */}
      <Grid container spacing={2}>
        <Grid item xs={6} sm={3}>
          <MetricsCard title="Avg Predicted Occupancy" value={avgPred} />
        </Grid>
        <Grid item xs={6} sm={3}>
          <MetricsCard
            title="Peak Occupancy"
            value={predValues.length ? Math.round(Math.max(...predValues)) : '-'}
            subtitle={preds[peakIdx]?.date || ''}
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <MetricsCard
            title="Min Occupancy"
            value={predValues.length ? Math.round(Math.min(...predValues)) : '-'}
            subtitle={preds[minIdx]?.date || ''}
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <MetricsCard title="MAE" value={metrics?.mae?.toFixed(1) || '-'} subtitle="Mean Absolute Error" />
        </Grid>
        {metrics?.mape != null && (
          <Grid item xs={6} sm={3}>
            <MetricsCard title="MAPE" value={`${metrics.mape.toFixed(1)}%`} subtitle="Mean Abs % Error" />
          </Grid>
        )}
        {metrics?.rmse != null && (
          <Grid item xs={6} sm={3}>
            <MetricsCard title="RMSE" value={metrics.rmse.toFixed(1)} subtitle="Root Mean Sq Error" />
          </Grid>
        )}
        {summary?.admission_stats?.total_records != null && (
          <Grid item xs={6} sm={3}>
            <MetricsCard title="Total Admissions" value={summary.admission_stats.total_records} />
          </Grid>
        )}
        {summary?.occupancy_stats?.total_records != null && (
          <Grid item xs={6} sm={3}>
            <MetricsCard title="Occupancy Records" value={summary.occupancy_stats.total_records} />
          </Grid>
        )}
      </Grid>
    </Box>
  );
}
