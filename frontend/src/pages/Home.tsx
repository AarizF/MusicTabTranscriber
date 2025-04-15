// filepath: [Home.tsx](http://_vscodecontentref_/3)
import React, { useState } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import FileUpload from '../components/FileUpload';

const Home: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    setLoading(true);
    setPdfUrl(null);

    try {
      const response = await fetch('/api/transcribe', {
        method: 'POST',
        body: new FormData().append('file', file),
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setPdfUrl(url);
      } else {
        console.error('Failed to transcribe audio');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        bgcolor: '#121212',
        color: '#fff',
      }}
    >
      <Typography variant="h4" gutterBottom>
        Drag MP3 Here
      </Typography>
      {loading ? (
        <CircularProgress color="secondary" />
      ) : pdfUrl ? (
        <a href={pdfUrl} download="transcription.pdf" style={{ color: '#90caf9' }}>
          Download PDF
        </a>
      ) : (
        <FileUpload onFileUpload={handleFileUpload} />
      )}
    </Box>
  );
};

export default Home;