/**
 * API Service Wrapper
 * Handles token storage, injected headers, and request logic.
 */

const API_BASE = import.meta.env.PROD
    ? 'https://doc-ai-backend-bpez.onrender.com/api'
    : '/api';

const api = {
    // Get token from storage
    getToken: () => localStorage.getItem('ultra_doc_token'),
    
    // Set token in storage
    setToken: (token) => localStorage.setItem('ultra_doc_token', token),
    
    // Remove token
    logout: () => localStorage.removeItem('ultra_doc_token'),

    // Authenticated request wrapper
    async request(endpoint, options = {}) {
        const token = this.getToken();
        
        const headers = {
            ...options.headers,
        };

        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        // Handle FormData vs JSON
        if (!(options.body instanceof FormData) && !headers['Content-Type']) {
            headers['Content-Type'] = 'application/json';
        }

        const response = await fetch(`${API_BASE}${endpoint}`, {
            ...options,
            headers
        });

        if (response.status === 401) {
            this.logout();
            window.location.reload(); // Force trigger login state
            throw new Error('Unauthorized');
        }

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(error.detail || 'Request failed');
        }

        return response.json();
    },

    // Specialized login method
    async login(username, password) {
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);

        let response;
        try {
            response = await fetch(`${API_BASE}/auth/login`, {
                method: 'POST',
                body: formData,
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
            });
        } catch (networkErr) {
            // Backend is offline or unreachable
            throw new Error('Cannot reach the server. Make sure the backend is running on port 8000.');
        }

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Login failed' }));
            throw new Error(error.detail || 'Login failed');
        }

        const data = await response.json();
        this.setToken(data.access_token);
        return data;
    },

    // Get current user
    async getCurrentUser() {
        return this.request('/auth/me');
    },

    // POST and consume a server-sent-event stream; calls onEvent(obj) for every `data:` frame
    async stream(endpoint, body, onEvent) {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${this.getToken()}` },
            body: JSON.stringify(body),
        });
        if (response.status === 401) {
            this.logout();
            window.location.reload();
            throw new Error('Unauthorized');
        }
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Request failed' }));
            throw new Error(error.detail || 'Request failed');
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let idx;
            while ((idx = buffer.indexOf('\n\n')) >= 0) {
                const frame = buffer.slice(0, idx);
                buffer = buffer.slice(idx + 2);
                if (frame.startsWith('data: ')) onEvent(JSON.parse(frame.slice(6)));
            }
        }
    },

    // URL of a rendered PDF page with an optional highlighted box (token in query: <img> can't send headers)
    pageUrl(docId, page, bbox) {
        const hl = bbox ? `&hl=${bbox.join(',')}` : '';
        return `${API_BASE}/rag/page/${docId}/${page}.png?token=${encodeURIComponent(this.getToken())}${hl}`;
    }
};

export default api;
