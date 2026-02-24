import React, { useState, useCallback } from 'react';
import {
  Box, Paper, Typography, Tabs, Tab, Button, Alert,
  LinearProgress, Table, TableHead, TableBody, TableRow,
  TableCell, TableContainer, Dialog, DialogTitle,
  DialogContent, DialogActions,
} from '@mui/material';
import { CloudUpload, Delete, Download } from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { uploadAdmissions, uploadHolidays, clearData } from '../services/api';

function TabPanel({ children, value, index }) {
  return value === index ? <Box sx={{ pt: 2 }}>{children}</Box> : null;
}

function FileDropzone({ onDrop, accept }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'text/csv': ['.csv'] }, maxFiles: 1,
  });
  return (
    <Box
      {...getRootProps()}
      sx={{
        border: '2px dashed', borderColor: isDragActive ? 'primary.main' : 'grey.400',
        borderRadius: 2, p: 4, textAlign: 'center', cursor: 'pointer',
        bgcolor: isDragActive ? 'action.hover' : 'background.paper',
        '&:hover': { bgcolor: 'action.hover' },
      }}
    >
      <input {...getInputProps()} />
      <CloudUpload sx={{ fontSize: 48, color: 'grey.500', mb: 1 }} />
      <Typography>{isDragActive ? 'Drop the file here...' : 'Drag & drop a CSV file, or click to browse'}</Typography>
    </Box>
  );
}

const SAMPLE_ADMISSIONS = `patient_id,admission_date,discharge_date,department,age_group,admission_type,primary_diagnosis_category
P001,2023-01-15 08:30:00,2023-01-20,Internal Medicine,51-65,Emergency,Infection
P002,2023-01-16 14:00:00,2023-01-18,Surgery,36-50,Elective,Appendectomy`;

const SAMPLE_HOLIDAYS = `date,holiday_name,is_public_holiday,is_school_holiday,region
2023-01-01,New Year's Day,true,false,US
2023-07-04,Independence Day,true,false,US`;

export default function DataUpload() {
  const [tab, setTab] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [preview, setPreview] = useState(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [clearType, setClearType] = useState('');

  const handleUpload = useCallback(async (files, type) => {
    if (!files.length) return;
    setUploading(true);
    setError('');
    setResult(null);

    // Preview first few rows
    const text = await files[0].text();
    const lines = text.split('\n').slice(0, 11);
    const headers = lines[0].split(',');
    const rows = lines.slice(1).filter(l => l.trim()).map(l => l.split(','));
    setPreview({ headers, rows });

    try {
      const res = type === 'admissions'
        ? await uploadAdmissions(files[0])
        : await uploadHolidays(files[0]);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed.');
    } finally {
      setUploading(false);
    }
  }, []);

  const handleClear = async () => {
    try {
      await clearData(clearType);
      setResult({ message: `${clearType} data cleared successfully.` });
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to clear data.');
    }
    setConfirmOpen(false);
  };

  const downloadSample = (content, filename) => {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>Upload Data</Typography>

      <Paper sx={{ p: 3 }}>
        <Tabs value={tab} onChange={(_, v) => { setTab(v); setResult(null); setError(''); setPreview(null); }}>
          <Tab label="Admission Data" />
          <Tab label="Holiday Data" />
        </Tabs>

        {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
        {result && <Alert severity="success" sx={{ mt: 2 }}>{result.message}</Alert>}
        {uploading && <LinearProgress sx={{ mt: 2 }} />}

        <TabPanel value={tab} index={0}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Required columns: patient_id, admission_date, discharge_date, department.
            Optional: age_group, admission_type, primary_diagnosis_category
          </Typography>
          <Button
            size="small" startIcon={<Download />} sx={{ mb: 2 }}
            onClick={() => downloadSample(SAMPLE_ADMISSIONS, 'sample_admissions.csv')}
          >
            Download Sample CSV
          </Button>
          <FileDropzone onDrop={(files) => handleUpload(files, 'admissions')} />
          <Button
            variant="outlined" color="error" startIcon={<Delete />} sx={{ mt: 2 }}
            onClick={() => { setClearType('admissions'); setConfirmOpen(true); }}
          >
            Clear Admission Data
          </Button>
        </TabPanel>

        <TabPanel value={tab} index={1}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Required columns: date, holiday_name.
            Optional: is_public_holiday, is_school_holiday, region
          </Typography>
          <Button
            size="small" startIcon={<Download />} sx={{ mb: 2 }}
            onClick={() => downloadSample(SAMPLE_HOLIDAYS, 'sample_holidays.csv')}
          >
            Download Sample CSV
          </Button>
          <FileDropzone onDrop={(files) => handleUpload(files, 'holidays')} />
          <Button
            variant="outlined" color="error" startIcon={<Delete />} sx={{ mt: 2 }}
            onClick={() => { setClearType('holidays'); setConfirmOpen(true); }}
          >
            Clear Holiday Data
          </Button>
        </TabPanel>

        {/* Preview table */}
        {preview && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle2" gutterBottom>Preview (first 10 rows)</Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    {preview.headers.map((h, i) => <TableCell key={i}><strong>{h}</strong></TableCell>)}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {preview.rows.map((row, ri) => (
                    <TableRow key={ri}>
                      {row.map((cell, ci) => <TableCell key={ci}>{cell}</TableCell>)}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}
      </Paper>

      {/* Confirmation dialog */}
      <Dialog open={confirmOpen} onClose={() => setConfirmOpen(false)}>
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>Are you sure you want to clear all {clearType} data? This cannot be undone.</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmOpen(false)}>Cancel</Button>
          <Button onClick={handleClear} color="error" variant="contained">Delete</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
