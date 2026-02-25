import React, { useState, useEffect } from 'react';
import {
  Box, Paper, Typography, Tabs, Tab, Grid, TextField, Slider,
  Button, Alert, CircularProgress, LinearProgress,
  Table, TableHead, TableBody, TableRow, TableCell, TableContainer,
  FormControl, InputLabel, Select, MenuItem, Card, CardContent,
} from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend,
} from 'chart.js';
import { trainOccupancyModel, trainLOSModel, listModels, activateModel } from '../services/api';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function TabPanel({ children, value, index }) {
  return value === index ? <Box sx={{ pt: 2 }}>{children}</Box> : null;
}

export default function ModelTraining() {
  const [tab, setTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [models, setModels] = useState([]);
  const [activating, setActivating] = useState(null);
  const [success, setSuccess] = useState('');

  // Occupancy model params
  const [occStart, setOccStart] = useState('2020-01-01');
  const [occEnd, setOccEnd] = useState('2023-12-31');
  const [occValidation, setOccValidation] = useState(0.2);
  const [occLayers, setOccLayers] = useState(2);
  const [occNodes, setOccNodes] = useState(10);
  const [occEpochs, setOccEpochs] = useState(500);
  const [occLR, setOccLR] = useState(0.001);

  // LOS model params
  const [losStart, setLosStart] = useState('2020-01-01');
  const [losEnd, setLosEnd] = useState('2023-12-31');
  const [losValidation, setLosValidation] = useState(0.2);
  const [losModelType, setLosModelType] = useState('random_forest');

  useEffect(() => {
    listModels().then(r => setModels(r.data.models)).catch(() => {});
  }, [result]);

  const handleTrainOccupancy = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await trainOccupancyModel({
        training_start: occStart,
        training_end: occEnd,
        validation_split: occValidation,
        hyperparameters: {
          hidden_layers: occLayers,
          nodes_per_layer: occNodes,
          epochs: occEpochs,
          learning_rate: occLR,
        },
      });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Training failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleTrainLOS = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await trainLOSModel({
        training_start: losStart,
        training_end: losEnd,
        validation_split: losValidation,
        model_type: losModelType,
      });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Training failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleActivate = async (modelId) => {
    setActivating(modelId);
    setSuccess('');
    setError('');
    try {
      const res = await activateModel(modelId);
      setSuccess(res.data?.message || `Model ${modelId} activated successfully.`);
      listModels().then(r => setModels(r.data.models)).catch(() => {});
    } catch (err) {
      setError(err.response?.data?.detail || 'Activation failed.');
    } finally {
      setActivating(null);
    }
  };

  // Loss curve chart
  const lossChart = result?.training_history ? {
    labels: result.training_history.loss.map((_, i) => i + 1),
    datasets: [
      {
        label: 'Training Loss',
        data: result.training_history.loss,
        borderColor: '#1565c0',
        borderWidth: 1.5,
        pointRadius: 0,
      },
      {
        label: 'Validation Loss',
        data: result.training_history.val_loss,
        borderColor: '#e53935',
        borderWidth: 1.5,
        pointRadius: 0,
      },
    ],
  } : null;

  return (
    <Box>
      <Typography variant="h5" gutterBottom>Model Training</Typography>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}

      <Paper sx={{ p: 3, mb: 3 }}>
        <Tabs value={tab} onChange={(_, v) => { setTab(v); setResult(null); setError(''); setSuccess(''); }}>
          <Tab label="Occupancy Model (NARX)" />
          <Tab label="LOS Model" />
        </Tabs>

        {loading && <LinearProgress sx={{ mt: 2 }} />}

        <TabPanel value={tab} index={0}>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <TextField label="Training Start" type="date" fullWidth
                InputLabelProps={{ shrink: true }}
                value={occStart} onChange={e => setOccStart(e.target.value)} />
            </Grid>
            <Grid item xs={6} sm={3}>
              <TextField label="Training End" type="date" fullWidth
                InputLabelProps={{ shrink: true }}
                value={occEnd} onChange={e => setOccEnd(e.target.value)} />
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography gutterBottom>Validation Split: {occValidation}</Typography>
              <Slider value={occValidation} onChange={(_, v) => setOccValidation(v)} min={0.05} max={0.5} step={0.05} />
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography gutterBottom>Hidden Layers: {occLayers}</Typography>
              <Slider value={occLayers} onChange={(_, v) => setOccLayers(v)} min={1} max={4} />
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography gutterBottom>Nodes/Layer: {occNodes}</Typography>
              <Slider value={occNodes} onChange={(_, v) => setOccNodes(v)} min={5} max={50} step={5} />
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography gutterBottom>Epochs: {occEpochs}</Typography>
              <Slider value={occEpochs} onChange={(_, v) => setOccEpochs(v)} min={100} max={2000} step={100} />
            </Grid>
            <Grid item xs={12}>
              <Button variant="contained" size="large" onClick={handleTrainOccupancy} disabled={loading}>
                {loading ? <CircularProgress size={24} color="inherit" /> : 'Train Occupancy Model'}
              </Button>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tab} index={1}>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <TextField label="Training Start" type="date" fullWidth
                InputLabelProps={{ shrink: true }}
                value={losStart} onChange={e => setLosStart(e.target.value)} />
            </Grid>
            <Grid item xs={6} sm={3}>
              <TextField label="Training End" type="date" fullWidth
                InputLabelProps={{ shrink: true }}
                value={losEnd} onChange={e => setLosEnd(e.target.value)} />
            </Grid>
            <Grid item xs={6} sm={3}>
              <FormControl fullWidth>
                <InputLabel>Model Type</InputLabel>
                <Select value={losModelType} label="Model Type" onChange={e => setLosModelType(e.target.value)}>
                  <MenuItem value="random_forest">Random Forest</MenuItem>
                  <MenuItem value="gradient_boosting">Gradient Boosting</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography gutterBottom>Validation Split: {losValidation}</Typography>
              <Slider value={losValidation} onChange={(_, v) => setLosValidation(v)} min={0.05} max={0.5} step={0.05} />
            </Grid>
            <Grid item xs={12}>
              <Button variant="contained" size="large" onClick={handleTrainLOS} disabled={loading}>
                {loading ? <CircularProgress size={24} color="inherit" /> : 'Train LOS Model'}
              </Button>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Training Results */}
        {result && (
          <Box sx={{ mt: 3 }}>
            <Alert severity="success" sx={{ mb: 2 }}>
              Model trained successfully! Version: {result.model_id}
            </Alert>
            <Grid container spacing={2}>
              {result.metrics && Object.entries(result.metrics).map(([key, val]) => (
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
            {lossChart && (
              <Box sx={{ mt: 3, height: 300 }}>
                <Line data={lossChart} options={{
                  responsive: true, maintainAspectRatio: false,
                  plugins: { title: { display: true, text: 'Training Loss Curves' } },
                  scales: { x: { title: { display: true, text: 'Epoch' } }, y: { title: { display: true, text: 'Loss' } } },
                }} />
              </Box>
            )}
          </Box>
        )}
      </Paper>

      {/* Model List */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>Available Models</Typography>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Version</TableCell>
                <TableCell>Created</TableCell>
                <TableCell>MAE</TableCell>
                <TableCell>RMSE</TableCell>
                <TableCell>Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {models.map((m, i) => (
                <TableRow key={i}>
                  <TableCell>{m.name}</TableCell>
                  <TableCell>{m.version}</TableCell>
                  <TableCell>{m.created_at}</TableCell>
                  <TableCell>{m.metrics?.mae?.toFixed(2) || '-'}</TableCell>
                  <TableCell>{m.metrics?.rmse?.toFixed(2) || '-'}</TableCell>
                  <TableCell>
                    <Button
                      size="small"
                      variant={m.is_active ? "contained" : "outlined"}
                      color={m.is_active ? "success" : "primary"}
                      disabled={activating === m.version}
                      onClick={() => handleActivate(m.version)}
                    >
                      {activating === m.version ? (
                        <CircularProgress size={18} color="inherit" />
                      ) : m.is_active ? 'Active' : 'Activate'}
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {models.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} align="center">No models trained yet.</TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </Box>
  );
}
