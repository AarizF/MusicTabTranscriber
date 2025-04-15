import React, { useState } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import FileUpload from '../components/FileUpload';

const Home: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    setLoading(true); // Set loading to true
    setPdfUrl(null); // Clear any previous PDF URL

    try {
      const formData = new FormData();
      formData.append('file', file);

      const backendUrl = process.env.REACT_APP_BACKEND_URL;

      const response = await fetch(`${backendUrl}/transcribe`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setPdfUrl(url); // Set the URL for the PDF file
      } else {
        console.error('Failed to transcribe audio');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
    } finally {
      setLoading(false); // Set loading to false after the request completes
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