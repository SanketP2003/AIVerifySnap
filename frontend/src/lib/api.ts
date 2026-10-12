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

export interface BulkDetectionResult {
    total_scanned: number;
    total_fake: number;
    total_real: number;
    fake_percentage: number;
    average_confidence: number;
    batch_elapsed_ms: number;
    results: DetectionResult[];
}

export interface ReverseSearchVisualMatch {
    title: string;
    link: string;
    source: string;
    thumbnail: string;
    similarity_score: number;
    match_type: string;
}

export interface ReverseSearchResult {
    filename: string;
    search_engine: string;
    visual_matches: ReverseSearchVisualMatch[];
    knowledge_graph: {
        title: string;
        subtitle: string;
        description: string;
        perceptual_hash?: string;
    };
    reverse_matches_count: number;
    processing_time_ms: number;
}

export interface IdentityProtectionMatch {
    platform: string;
    url: string;
    match_type: string;
    similarity_score: number;
    date_spotted: string;
    status: string;
    infringement_risk: string;
}

export interface IdentityProtectionResult {
    filename: string;
    biometric_hash: string;
    timestamp: string;
    infringement_risk_score: number;
    threat_level: string;
    total_infringements_found: number;
    detected_matches: IdentityProtectionMatch[];
    takedown_eligible: boolean;
    dmca_notice_template: string;
    detection_analysis?: DetectionResult;
}

export interface DetectionHistoryItem {
    scanId: number;
    imagePath: string;
    resultLabel: string;
    confidenceScore: number;
    analysisMetadata: string;
    scanTimestamp: string;
    user: { id: number; name: string; email: string; plan?: string } | null;
}

export interface UserQuota {
    plan: 'STARTER' | 'PROFESSIONAL' | 'ENTERPRISE';
    scansToday: number;
    dailyLimit: number; // -1 for unlimited
    remainingToday: number | string;
    historyDaysLimit: number; // -1 for full
    bulkLimit: number; // 1 for Starter, 50 for Pro, -1 for Enterprise
    identityProtectionAllowed: boolean;
    pdfExportAllowed: boolean;
    apiAccessAllowed?: boolean;
}

// Client-side stored plan helper for interactive testing & active user tier
export const planStorage = {
    getPlan: (): 'STARTER' | 'PROFESSIONAL' | 'ENTERPRISE' => {
        if (typeof window === 'undefined') return 'PROFESSIONAL';
        return (localStorage.getItem('avs_user_plan') as 'STARTER' | 'PROFESSIONAL' | 'ENTERPRISE') || 'PROFESSIONAL';
    },
    setPlan: (plan: 'STARTER' | 'PROFESSIONAL' | 'ENTERPRISE') => {
        if (typeof window !== 'undefined') {
            localStorage.setItem('avs_user_plan', plan);
            window.dispatchEvent(new Event('avs_plan_changed'));
        }
    }
};

export const detectionApi = {
    detectImage: async (file: File, userId?: number): Promise<DetectionResult> => {
        const formData = new FormData();
        formData.append('file', file);
        if (userId !== undefined) {
            formData.append('userId', userId.toString());
        }
        const response = await api.post('/detection/analyze', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return response.data;
    },

    detectBulk: async (files: File[], userId?: number): Promise<BulkDetectionResult> => {
        const formData = new FormData();
        files.forEach((file) => formData.append('files', file));
        if (userId !== undefined) {
            formData.append('userId', userId.toString());
        }
        const response = await api.post('/detection/analyze-bulk', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return response.data;
    },

    reverseSearch: async (file: File): Promise<ReverseSearchResult> => {
        const formData = new FormData();
        formData.append('file', file);
        const response = await api.post('/detection/reverse-search', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return response.data;
    },

    identityProtect: async (file: File, userId?: number): Promise<IdentityProtectionResult> => {
        const formData = new FormData();
        formData.append('file', file);
        if (userId !== undefined) {
            formData.append('userId', userId.toString());
        }
        const response = await api.post('/detection/identity-protect', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return response.data;
    },

    getHistory: async (userId?: number): Promise<DetectionHistoryItem[]> => {
        const url = userId ? `/detection/history?userId=${userId}` : '/detection/history';
        const response = await api.get(url);
        return response.data;
    },

    getReport: async (id: string): Promise<DetectionHistoryItem> => {
        const response = await api.get(`/detection/history/${id}`);
        return response.data;
    },

    deleteHistoryItem: async (id: number): Promise<{ message: string; scanId: string }> => {
        const response = await api.delete(`/detection/history/${id}`);
        return response.data;
    },

    registerUser: async (userData: { name: string; email: string; passwordHash?: string; role?: string; clerkId?: string; plan?: string }) => {
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

    updateUserPlan: async (identifier: string, plan: 'STARTER' | 'PROFESSIONAL' | 'ENTERPRISE') => {
        planStorage.setPlan(plan);
        try {
            const response = await api.put(`/api/users/${identifier}/plan`, { plan });
            return response.data;
        } catch {
            return { plan };
        }
    },

    getUserQuota: async (identifier: string): Promise<UserQuota> => {
        try {
            const response = await api.get(`/api/users/${identifier}/quota`);
            return response.data;
        } catch {
            // Local fallback based on stored plan
            const activePlan = planStorage.getPlan();
            return {
                plan: activePlan,
                scansToday: 2,
                dailyLimit: activePlan === 'STARTER' ? 10 : -1,
                remainingToday: activePlan === 'STARTER' ? 8 : 'Unlimited',
                historyDaysLimit: activePlan === 'STARTER' ? 7 : -1,
                bulkLimit: activePlan === 'STARTER' ? 1 : activePlan === 'PROFESSIONAL' ? 50 : -1,
                identityProtectionAllowed: activePlan !== 'STARTER',
                pdfExportAllowed: activePlan !== 'STARTER',
                apiAccessAllowed: activePlan === 'ENTERPRISE'
            };
        }
    },

    generateApiKey: async (identifier: string) => {
        const response = await api.post(`/api/users/${identifier}/api-key`);
        return response.data;
    }
};
