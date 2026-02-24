import React, { useState, useEffect } from 'react';
import {
  Box, Paper, Typography, Tabs, Tab, Grid, Alert,
  CircularProgress, Card, CardContent, Select, MenuItem,
  FormControl, InputLabel,
} from '@mui/material';
import { Bar, Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, BarElement, Title, Tooltip, Legend,
} from 'chart.js';
import {
  getForecastAccuracy, getSeasonalPatterns,
  getCapacityPlanning, getFeatureImportance,
} from '../services/api';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend);

function TabPanel({ children, value, index }) {
  return value === index ? <Box sx={{ pt: 2 }}>{children}</Box> : null;
}

export default function Analytics() {
  const [tab, setTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [accuracy, setAccuracy] = useState(null);
  const [seasonal, setSeasonal] = useState(null);
  const [capacity, setCapacity] = useState(null);
  const [features, setFeatures] = useState(null);
  const [scenario, setScenario] = useState('realistic');

  useEffect(() => {
    setLoading(true);
    setError('');
    const loaders = [
      getForecastAccuracy({}).then(r => setAccuracy(r.data)).catch(() => {}),
      getSeasonalPatterns().then(r => setSeasonal(r.data)).catch(() => {}),
      getCapacityPlanning({ scenario }).then(r => setCapacity(r.data)).catch(() => {}),
      getFeatureImportance().then(r => setFeatures(r.data)).catch(() => {}),
    ];
    Promise.all(loaders).finally(() => setLoading(false));
  }, [scenario]);

  const accuracyChart = accuracy ? {
    labels: accuracy.dates,
    datasets: [
      { label: 'Predicted', data: accuracy.predicted, borderColor: '#1565c0', borderWidth: 2, pointRadius: 1 },
      { label: 'Actual', data: accuracy.actual, borderColor: '#e53935', borderWidth: 2, pointRadius: 1 },
    ],
  } : null;

  const monthlyChart = seasonal?.monthly_averages ? {
    labels: seasonal.monthly_averages.map(m => m.month),
    datasets: [{
      label: 'Mean Occupancy',
      data: seasonal.monthly_averages.map(m => m.mean_occupancy),
      backgroundColor: 'rgba(21,101,192,0.6)',
    }],
  } : null;

  const dowChart = seasonal?.day_of_week_patterns ? {
    labels: seasonal.day_of_week_patterns.map(d => d.day),
    datasets: [{
      label: 'Mean Occupancy',
      data: seasonal.day_of_week_patterns.map(d => d.mean_occupancy),
      backgroundColor: 'rgba(0,131,143,0.6)',
    }],
  } : null;

  const featureChart = features?.occupancy_features ? {
    labels: features.occupancy_features.map(f => f.name),
    datasets: [{
      label: 'Importance',
      data: features.occupancy_features.map(f => f.importance),
      backgroundColor: 'rgba(21,101,192,0.7)',
    }],
  } : null;

  const losFeatureChart = features?.los_features?.length ? {
    labels: features.los_features.map(f => f.feature_name),
    datasets: [{
      label: 'Importance',
      data: features.los_features.map(f => f.importance_score),
      backgroundColor: 'rgba(0,131,143,0.7)',
    }],
  } : null;

  return (
    <Box>
      <Typography variant="h5" gutterBottom>Analytics</Typography>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      {loading && <CircularProgress sx={{ display: 'block', mx: 'auto', my: 4 }} />}

      <Paper sx={{ p: 3 }}>
        <Tabs value={tab} onChange={(_, v) => setTab(v)}>
          <Tab label="Forecast Accuracy" />
          <Tab label="Seasonal Patterns" />
          <Tab label="Capacity Planning" />
          <Tab label="Feature Importance" />
        </Tabs>

        {/* Forecast Accuracy */}
        <TabPanel value={tab} index={0}>
          {accuracyChart ? (
            <>
              <Box sx={{ height: 400 }}>
                <Line data={accuracyChart} options={{
                  responsive: true, maintainAspectRatio: false,
                  plugins: { title: { display: true, text: 'Predicted vs Actual Occupancy' } },
                  scales: {
                    x: { ticks: { maxTicksLimit: 15 } },
                    y: { title: { display: true, text: 'Beds' } },
                  },
                }} />
              </Box>
              {accuracy.metrics && (
                <Grid container spacing={2} sx={{ mt: 2 }}>
                  {Object.entries(accuracy.metrics).map(([key, val]) => (
                    <Grid item xs={6} sm={3} key={key}>
                      <Card>
                        <CardContent>
                          <Typography variant="body2" color="text.secondary">{key.toUpperCase()}</Typography>
                          <Typography variant="h5">{typeof val === 'number' ? val.toFixed(2) : val}</Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              )}
            </>
          ) : (
            <Typography color="text.secondary" sx={{ mt: 2 }}>
              No forecast accuracy data available. Generate a forecast first.
            </Typography>
          )}
        </TabPanel>

        {/* Seasonal Patterns */}
        <TabPanel value={tab} index={1}>
          <Grid container spacing={3}>
            {monthlyChart && (
              <Grid item xs={12} md={6}>
                <Box sx={{ height: 300 }}>
                  <Bar data={monthlyChart} options={{
                    responsive: true, maintainAspectRatio: false,
                    plugins: { title: { display: true, text: 'Monthly Occupancy Averages' } },
                  }} />
                </Box>
              </Grid>
            )}
            {dowChart && (
              <Grid item xs={12} md={6}>
                <Box sx={{ height: 300 }}>
                  <Bar data={dowChart} options={{
                    responsive: true, maintainAspectRatio: false,
                    plugins: { title: { display: true, text: 'Day of Week Patterns' } },
                  }} />
                </Box>
              </Grid>
            )}
            {seasonal?.holiday_impact && (
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6">Holiday Impact on LOS</Typography>
                    <Typography>Weekday avg LOS: {seasonal.holiday_impact.weekday_avg_los} days</Typography>
                    <Typography>Weekend avg LOS: {seasonal.holiday_impact.weekend_avg_los} days</Typography>
                  </CardContent>
                </Card>
              </Grid>
            )}
          </Grid>
        </TabPanel>

        {/* Capacity Planning */}
        <TabPanel value={tab} index={2}>
          <FormControl sx={{ mb: 3, minWidth: 200 }}>
            <InputLabel>Scenario</InputLabel>
            <Select value={scenario} label="Scenario" onChange={e => setScenario(e.target.value)}>
              <MenuItem value="optimistic">Optimistic</MenuItem>
              <MenuItem value="realistic">Realistic</MenuItem>
              <MenuItem value="pessimistic">Pessimistic</MenuItem>
            </Select>
          </FormControl>
          {capacity ? (
            <Grid container spacing={2}>
              <Grid item xs={6} sm={3}>
                <Card>
                  <CardContent>
                    <Typography variant="body2" color="text.secondary">Recommended Beds</Typography>
                    <Typography variant="h4">{capacity.recommended_beds}</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Card>
                  <CardContent>
                    <Typography variant="body2" color="text.secondary">Utilization Forecast</Typography>
                    <Typography variant="h4">{capacity.utilization_forecast}%</Typography>
                  </CardContent>
                </Card>
              </Grid>
              {capacity.staffing_recommendations && (
                <Grid item xs={6} sm={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="body2" color="text.secondary">Nurses/Shift</Typography>
                      <Typography variant="h4">{capacity.staffing_recommendations.nurses_per_shift}</Typography>
                    </CardContent>
                  </Card>
                </Grid>
              )}
              {capacity.risk_assessment && (
                <Grid item xs={6} sm={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="body2" color="text.secondary">Risk Level</Typography>
                      <Typography variant="h4" color={
                        capacity.risk_assessment.risk_level === 'High' ? 'error.main' :
                        capacity.risk_assessment.risk_level === 'Medium' ? 'warning.main' : 'success.main'
                      }>
                        {capacity.risk_assessment.risk_level}
                      </Typography>
                      <Typography variant="body2">
                        Peak: {capacity.risk_assessment.peak_occupancy} |
                        Overflow: {capacity.risk_assessment.overflow_probability}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
          ) : (
            <Typography color="text.secondary">No data available for capacity planning.</Typography>
          )}
        </TabPanel>

        {/* Feature Importance */}
        <TabPanel value={tab} index={3}>
          <Grid container spacing={3}>
            {featureChart && (
              <Grid item xs={12} md={6}>
                <Box sx={{ height: 350 }}>
                  <Bar data={featureChart} options={{
                    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                    plugins: { title: { display: true, text: 'Occupancy Model Features' } },
                  }} />
                </Box>
              </Grid>
            )}
            {losFeatureChart && (
              <Grid item xs={12} md={6}>
                <Box sx={{ height: 350 }}>
                  <Bar data={losFeatureChart} options={{
                    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                    plugins: { title: { display: true, text: 'LOS Model Features' } },
                  }} />
                </Box>
              </Grid>
            )}
            {!featureChart && !losFeatureChart && (
              <Grid item xs={12}>
                <Typography color="text.secondary">
                  Train models to see feature importance analysis.
                </Typography>
              </Grid>
            )}
          </Grid>
        </TabPanel>
      </Paper>
    </Box>
  );
}
