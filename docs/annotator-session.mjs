export class SessionExpiredError extends Error {}

export class AnnotatorSession extends EventTarget {
    constructor(baseUrl) {
        super();
        this.baseUrl = baseUrl.replace(/\/$/, "");
        this.session = null;
    }

    get authenticated() {
        return Boolean(this.session && Date.parse(this.session.expires_at) > Date.now());
    }

    logout() {
        this.session = null;
        this.dispatchEvent(new Event("logout"));
    }

    async login(password) {
        const response = await fetch(`${this.baseUrl}/auth/login`, {
            method: "POST", credentials: "omit", cache: "no-store",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ password }),
        });
        if (!response.ok) {
            throw new Error(response.status === 401 ? "Incorrect password." :
                response.status === 503 ? "Login service unavailable. Please try again later." : "Unable to sign in.");
        }
        const data = await response.json();
        if (!data.access_token || data.token_type !== "Bearer" || !(Date.parse(data.expires_at) > Date.now())) {
            throw new Error("Invalid login response.");
        }
        this.session = data;
    }

    async request(path, options = {}) {
        if (!/^\/(species|references|candidates|annotations)(\?|$)/.test(path)) {
            throw new Error("Unsupported API route.");
        }
        if (!this.authenticated) return this.expired();

        const headers = new Headers(options.headers);
        headers.set("Authorization", `Bearer ${this.session.access_token}`);
        const session = this.session;
        const response = await fetch(`${this.baseUrl}${path}`, {
            ...options, headers, credentials: "omit", cache: "no-store",
        });
        if (response.status === 401 && this.session === session) return this.expired();
        return response;
    }

    expired() {
        this.logout();
        throw new SessionExpiredError("Annotator session expired. Enter the password again.");
    }
}
