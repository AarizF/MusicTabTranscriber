import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box } from '@mui/material';

interface FileUploadProps {
  onFileUpload: (file: File) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload }) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileUpload(acceptedFiles[0]);
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'audio/mpeg': ['.mp3'] }, // Correct MIME type mapping
  });

  return (
    <Box
      {...getRootProps()}
      sx={{
        width: '300px',
        height: '200px',
        border: '2px dashed #90caf9',
        borderRadius: '8px',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        cursor: 'pointer',
        bgcolor: isDragActive ? '#1e88e5' : '#1c1c1c',
        color: '#90caf9',
      }}
    >
      <input {...getInputProps()} />
      {isDragActive ? 'Drop the file here...' : 'Drag and drop an MP3 file here'}
    </Box>
  );
};

export default FileUpload;