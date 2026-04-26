import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

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
    details: {
        raw_output?: Array<{ label: string; score: number }>;
        ela_mean?: number;
        ela_std?: number;
        ela_max?: number;
        ela_image_base64?: string;
        [key: string]: unknown;
    };
}

export interface DetectionHistoryItem {
    scanId: number;
    imagePath: string;
    resultLabel: string;
    confidenceScore: number;
    analysisMetadata: string;
    scanTimestamp: string;
    user: { id: number; name: string; email: string } | null;
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

    getHistory: async (): Promise<DetectionHistoryItem[]> => {
        const response = await api.get('/detection/history');
        return response.data;
    },

    getReport: async (id: string): Promise<DetectionHistoryItem> => {
        const response = await api.get(`/detection/history/${id}`);
        return response.data;
    },

    registerUser: async (userData: { name: string; email: string; passwordHash?: string; role?: string; clerkId?: string }) => {
        const response = await api.post('/api/users/register', userData);
        return response.data;
    },

    getUserByName: async (name: string) => {
        const response = await api.get(`/api/users/${name}`);
        return response.data;
    },

    syncClerkUser: async (clerkData: { clerkId: string; name: string; email: string }) => {
        const response = await api.post('/api/users/clerk-sync', clerkData);
        return response.data;
    },
};
