import axios from 'axios';

const API_BASE_URL = 'http://localhost:8080';

const api = axios.create({
    baseURL: API_BASE_URL,
});

export interface DetectionResult {
    filename: string;
    prediction: string;
    is_deepfake: boolean;
    confidence: number;
    raw_score: number;
    processing_time_ms: number;
    elapsed_ms: number;
    model_status: string;
    details: Record<string, unknown>;
}

export const detectionApi = {
    detectImage: async (file: File, userId?: number): Promise<DetectionResult> => {
        const formData = new FormData();
        formData.append('file', file);
        if (userId !== undefined) {
            formData.append('userId', userId.toString());
        }
        const response = await api.post('/detection/analyze', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    getHistory: async () => {
        const response = await api.get('/detection/history');
        return response.data;
    },

    getReport: async (id: string) => {
        const response = await api.get(`/detection/history/${id}`);
        return response.data;
    },

    createUser: async (userData: Record<string, unknown>) => {
        const response = await api.post('/api/users', userData);
        return response.data;
    },
};
