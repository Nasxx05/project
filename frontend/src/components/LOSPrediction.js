import React, { useState, useCallback } from 'react';
import {
  Box, Grid, Paper, Typography, Button, Select, MenuItem,
  FormControl, InputLabel, TextField, Alert, CircularProgress,
  Card, CardContent, Divider, Chip, LinearProgress,
} from '@mui/material';
import { CloudUpload } from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { predictLOS, batchPredictLOS } from '../services/api';

const AGE_GROUPS = ['0-17', '18-35', '36-50', '51-65', '65+'];
const ADMISSION_TYPES = ['Emergency', 'Elective', 'Transfer'];
const DEPARTMENTS = [
  'Internal Medicine', 'Surgery', 'Cardiology', 'Respiratory',
  'Neurology', 'Oncology', 'Orthopedics', 'Pediatrics',
];
const DIAGNOSES = [
  'Diabetes', 'Infection', 'Renal', 'GI Disorder',
  'MI', 'Heart Failure', 'Pneumonia', 'COPD',
  'Stroke', 'Chemotherapy', 'Hip Fracture', 'Appendectomy',
];

export default function LOSPrediction() {
  const [form, setForm] = useState({
    age_group: '51-65',
    admission_type: 'Emergency',
    department: 'Internal Medicine',
    diagnosis_category: 'Infection',
    admission_date: '',
    admission_hour: 14,
    is_holiday_period: false,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [batchResult, setBatchResult] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);

  const handleChange = (field) => (e) => {
    setForm(prev => ({ ...prev, [field]: e.target.value }));
  };

  const handlePredict = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await predictLOS({
        age_group: form.age_group,
        admission_type: form.admission_type,
        department: form.department,
        diagnosis_category: form.diagnosis_category,
        admission_hour: parseInt(form.admission_hour, 10),
        is_holiday_period: form.is_holiday_period,
      });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Ensure the LOS model is trained.');
    } finally {
      setLoading(false);
    }
  }, [form]);

  const handleBatchUpload = useCallback(async (files) => {
    if (!files.length) return;
    setBatchLoading(true);
    setBatchResult(null);
    try {
      const res = await batchPredictLOS(files[0]);
      setBatchResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Batch prediction failed.');
    } finally {
      setBatchLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop: handleBatchUpload,
    accept: { 'text/csv': ['.csv'] },
    maxFiles: 1,
  });

  return (
    <Box>
      <Typography variant="h5" gutterBottom>Length of Stay Prediction</Typography>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      <Grid container spacing={3}>
        {/* Input Form */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Patient Information</Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Age Group</InputLabel>
                  <Select value={form.age_group} label="Age Group" onChange={handleChange('age_group')}>
                    {AGE_GROUPS.map(g => <MenuItem key={g} value={g}>{g}</MenuItem>)}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Admission Type</InputLabel>
                  <Select value={form.admission_type} label="Admission Type" onChange={handleChange('admission_type')}>
                    {ADMISSION_TYPES.map(t => <MenuItem key={t} value={t}>{t}</MenuItem>)}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Department</InputLabel>
                  <Select value={form.department} label="Department" onChange={handleChange('department')}>
                    {DEPARTMENTS.map(d => <MenuItem key={d} value={d}>{d}</MenuItem>)}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Diagnosis Category</InputLabel>
                  <Select value={form.diagnosis_category} label="Diagnosis Category" onChange={handleChange('diagnosis_category')}>
                    {DIAGNOSES.map(d => <MenuItem key={d} value={d}>{d}</MenuItem>)}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Admission Date" type="date" fullWidth
                  InputLabelProps={{ shrink: true }}
                  value={form.admission_date} onChange={handleChange('admission_date')}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Admission Hour (0-23)" type="number" fullWidth
                  inputProps={{ min: 0, max: 23 }}
                  value={form.admission_hour} onChange={handleChange('admission_hour')}
                />
              </Grid>
              <Grid item xs={12}>
                <Button
                  variant="contained" fullWidth size="large"
                  onClick={handlePredict} disabled={loading}
                >
                  {loading ? <CircularProgress size={24} color="inherit" /> : 'Predict Length of Stay'}
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Results */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>Prediction Results</Typography>
            {result ? (
              <Box>
                <Box sx={{ textAlign: 'center', my: 3 }}>
                  <Typography variant="h2" color="primary">{result.predicted_los}</Typography>
                  <Typography variant="h6" color="text.secondary">days (predicted)</Typography>
                </Box>
                <Divider sx={{ my: 2 }} />
                <Box sx={{ textAlign: 'center', mb: 2 }}>
                  <Typography variant="body1">
                    90% Confidence Interval
                  </Typography>
                  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 2, mt: 1 }}>
                    <Chip label={`${result.lower_bound} days`} color="info" variant="outlined" />
                    <Typography variant="h6">-</Typography>
                    <Chip label={`${result.upper_bound} days`} color="info" variant="outlined" />
                  </Box>
                </Box>
                {/* Visual interval bar */}
                <Box sx={{ mx: 4, mt: 2 }}>
                  <Box sx={{ position: 'relative', height: 30, bgcolor: 'grey.200', borderRadius: 1 }}>
                    <Box sx={{
                      position: 'absolute',
                      left: `${Math.max(0, (result.lower_bound / (result.upper_bound * 1.2)) * 100)}%`,
                      width: `${((result.upper_bound - result.lower_bound) / (result.upper_bound * 1.2)) * 100}%`,
                      height: '100%', bgcolor: 'primary.light', borderRadius: 1, opacity: 0.5,
                    }} />
                    <Box sx={{
                      position: 'absolute',
                      left: `${(result.predicted_los / (result.upper_bound * 1.2)) * 100}%`,
                      width: 4, height: '100%', bgcolor: 'primary.main', borderRadius: 1,
                    }} />
                  </Box>
                </Box>
                <Divider sx={{ my: 2 }} />
                <Typography variant="body2" color="text.secondary">
                  Similar cases in database: {result.similar_cases}
                </Typography>
              </Box>
            ) : (
              <Typography color="text.secondary" sx={{ textAlign: 'center', mt: 4 }}>
                Fill in patient information and click "Predict" to see results.
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Batch Prediction */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Batch Prediction</Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Upload a CSV file with patient characteristics for bulk LOS predictions.
            </Typography>
            <Box
              {...getRootProps()}
              sx={{
                border: '2px dashed', borderColor: 'grey.400', borderRadius: 2,
                p: 3, textAlign: 'center', cursor: 'pointer',
                '&:hover': { bgcolor: 'action.hover' },
              }}
            >
              <input {...getInputProps()} />
              <CloudUpload sx={{ fontSize: 36, color: 'grey.500' }} />
              <Typography>Drop CSV file here or click to browse</Typography>
            </Box>
            {batchLoading && <LinearProgress sx={{ mt: 2 }} />}
            {batchResult && (
              <Box sx={{ mt: 2 }}>
                <Alert severity="success">
                  Processed {batchResult.summary_statistics?.count || 0} patients.
                  Mean predicted LOS: {batchResult.summary_statistics?.mean_los || '-'} days.
                </Alert>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
