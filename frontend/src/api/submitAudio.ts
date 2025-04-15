// filepath: [submitAudio.ts](http://_vscodecontentref_/5)
import axios from 'axios';

export const submitAudio = async (file: File): Promise<Blob> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post('/api/transcribe', formData, {
    responseType: 'blob',
  });

  return response.data;
};