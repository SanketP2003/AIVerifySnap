"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import {
    Cpu,
    Sliders,
    Moon,
    Sun,
    Monitor,
    CheckCircle2,
    Save,
    RotateCcw,
    User,
    ArrowUpRight
} from "lucide-react";
import { useTheme } from "next-themes";
import { useUser, useAuth } from "@clerk/nextjs";

import { detectionApi, HealthStatus } from "@/lib/api";

interface WorkspacePreferences {
    sensitivity: number;
    elaFactor: number;
    autoCertificate: boolean;
    hashLedger: boolean;
    tableDensity: "comfortable" | "compact";
}

const STORAGE_KEY = "avs_workspace_preferences";

export default function WorkspaceSettingsPage() {
    const { theme, setTheme } = useTheme();
    const { user, isLoaded: userLoaded } = useUser();
    const { isSignedIn } = useAuth();

    // Forensic Engine Preferences
    const [sensitivity, setSensitivity] = useState(50);
    const [elaFactor, setElaFactor] = useState(15);
    const [autoCertificate, setAutoCertificate] = useState(true);
    const [hashLedger, setHashLedger] = useState(true);
    const [tableDensity, setTableDensity] = useState<"comfortable" | "compact">("comfortable");
    const [savedMessage, setSavedMessage] = useState<string | null>(null);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [saving, setSaving] = useState(false);

    // Diagnostics State
    const [backendStatus, setBackendStatus] = useState<"online" | "checking" | "offline">("checking");
    const [backendPing, setBackendPing] = useState<number | null>(null);
    const [healthDetails, setHealthDetails] = useState<HealthStatus | null>(null);

    // Load persisted preferences on mount
    useEffect(() => {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored) {
                const parsed: Partial<WorkspacePreferences> = JSON.parse(stored);
                if (typeof parsed.sensitivity === "number") setSensitivity(parsed.sensitivity);
                if (typeof parsed.elaFactor === "number") setElaFactor(parsed.elaFactor);
                if (typeof parsed.autoCertificate === "boolean") setAutoCertificate(parsed.autoCertificate);
                if (typeof parsed.hashLedger === "boolean") setHashLedger(parsed.hashLedger);
                if (parsed.tableDensity === "comfortable" || parsed.tableDensity === "compact") {
                    setTableDensity(parsed.tableDensity);
                }
            }
        } catch (e) {
            console.error("Failed to load local preferences:", e);
        }

        // Live health check using configured API
        detectionApi.getHealth()
            .then((h) => {
                setHealthDetails(h);
                if (h.status === "UP") {
                    setBackendStatus("online");
                    setBackendPing(h.latencyMs ?? null);
                } else {
                    setBackendStatus("offline");
                    setBackendPing(null);
                }
            })
            .catch(() => {
                setBackendStatus("offline");
                setBackendPing(null);
            });
    }, []);

    const handleSave = () => {
        setSaving(true);
        setErrorMessage(null);
        try {
            const prefs: WorkspacePreferences = {
                sensitivity,
                elaFactor,
                autoCertificate,
                hashLedger,
                tableDensity,
            };
            localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
            setSavedMessage("Workspace preferences saved to browser storage successfully.");
            setTimeout(() => setSavedMessage(null), 3500);
        } catch (err: unknown) {
            setErrorMessage(err instanceof Error ? err.message : "Failed to save preferences to storage");
            setTimeout(() => setErrorMessage(null), 4000);
        } finally {
            setSaving(false);
        }
    };

    const handleReset = () => {
        setSensitivity(50);
        setElaFactor(15);
        setAutoCertificate(true);
        setHashLedger(true);
        setTableDensity("comfortable");
        try {
            localStorage.removeItem(STORAGE_KEY);
        } catch {
            // ignore
        }
        setSavedMessage("Preferences reset to platform defaults.");
        setTimeout(() => setSavedMessage(null), 3000);
    };

    return (
        <div className="p-4 sm:p-6 md:p-8 max-w-6xl mx-auto space-y-8">
            {/* Header */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-6 border-b border-border/80 gap-4">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span className={`w-2 h-2 rounded-full ${backendStatus === "online" ? "bg-green-500 animate-pulse" : "bg-primary"}`} />
                        <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground font-mono">
                            Configuration &middot; Forensic Cluster
                        </span>
                    </div>
                    <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-foreground font-sans">
                        Workspace Settings
                    </h1>
                    <p className="text-xs sm:text-sm text-muted-foreground mt-0.5">
                        Fine-tune forensic engine parameters, inspect node diagnostics, and manage session preferences.
                    </p>
                </div>

                <div className="flex items-center gap-2.5">
                    <button
                        type="button"
                        onClick={handleReset}
                        className="px-4 py-2 rounded-xl bg-card border border-border text-xs font-semibold text-foreground hover:bg-muted transition-colors flex items-center gap-1.5 shadow-sm"
                    >
                        <RotateCcw className="w-3.5 h-3.5 text-muted-foreground" />
                        <span>Reset Defaults</span>
                    </button>
                    <button
                        type="button"
                        onClick={handleSave}
                        disabled={saving}
                        className="px-5 py-2 rounded-xl bg-foreground text-background text-xs font-bold uppercase tracking-wider hover:bg-foreground/90 transition-all shadow-md flex items-center gap-2"
                    >
                        <Save className="w-3.5 h-3.5" />
                        <span>{saving ? "Saving..." : "Save Preferences"}</span>
                    </button>
                </div>
            </div>

            {savedMessage && (
                <div className="p-4 rounded-2xl bg-green-500/10 border border-green-500/30 text-green-700 dark:text-green-400 text-xs font-semibold flex items-center gap-2 transition-all">
                    <CheckCircle2 className="w-4 h-4 shrink-0" />
                    <span>{savedMessage}</span>
                </div>
            )}

            {errorMessage && (
                <div className="p-4 rounded-2xl bg-destructive/10 border border-destructive/30 text-destructive text-xs font-semibold flex items-center gap-2 transition-all">
                    <span>{errorMessage}</span>
                </div>
            )}

            {/* Grid Sections */}
            <div className="grid lg:grid-cols-12 gap-8">
                {/* Left (8 cols): Engine Config & Appearance */}
                <div className="lg:col-span-8 space-y-6">
                    {/* Section 1: Forensic Detection Sensitivity */}
                    <div className="p-6 rounded-2xl bg-card border border-border/80 shadow-sm space-y-6">
                        <div className="flex items-center gap-3">
                            <div className="p-2.5 rounded-xl bg-primary/10 text-primary border border-primary/20">
                                <Sliders className="w-5 h-5" />
                            </div>
                            <div>
                                <h2 className="text-base font-bold text-foreground font-sans">
                                    Forensic Pipeline Parameters
                                </h2>
                                <p className="text-xs text-muted-foreground">
                                    Calibrate model confidence threshold and Error Level Analysis multipliers
                                </p>
                            </div>
                        </div>

                        <div className="space-y-6 pt-2">
                            {/* Sensitivity Slider */}
                            <div className="space-y-2">
                                <div className="flex items-center justify-between text-xs font-semibold">
                                    <label htmlFor="sensitivity-range" className="text-foreground">
                                        Synthetic Verdict Threshold
                                    </label>
                                    <span className="font-mono text-primary font-bold">{sensitivity}% probability</span>
                                </div>
                                <input
                                    id="sensitivity-range"
                                    type="range"
                                    min="30"
                                    max="80"
                                    value={sensitivity}
                                    onChange={(e) => setSensitivity(Number(e.target.value))}
                                    className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-foreground"
                                />
                                <div className="flex justify-between text-[11px] text-muted-foreground">
                                    <span>Aggressive (30%)</span>
                                    <span>Balanced (50% Standard)</span>
                                    <span>Cautious (80%)</span>
                                </div>
                            </div>

                            {/* ELA Rescaling Factor */}
                            <div className="space-y-2 pt-2 border-t border-border/50">
                                <div className="flex items-center justify-between text-xs font-semibold">
                                    <label htmlFor="ela-range" className="text-foreground">
                                        ELA Deviation Amplification Multiplier
                                    </label>
                                    <span className="font-mono text-primary font-bold">{elaFactor}x</span>
                                </div>
                                <input
                                    id="ela-range"
                                    type="range"
                                    min="5"
                                    max="30"
                                    value={elaFactor}
                                    onChange={(e) => setElaFactor(Number(e.target.value))}
                                    className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-foreground"
                                />
                                <p className="text-[11px] text-muted-foreground">
                                    Scales the pixel difference between the original and 90% recompressed frame for visual highlighting.
                                </p>
                            </div>

                            {/* Checkbox toggles */}
                            <div className="space-y-3 pt-2 border-t border-border/50">
                                <label className="flex items-center justify-between p-3 rounded-2xl bg-muted/30 border border-border/40 cursor-pointer hover:bg-muted/50 transition-colors">
                                    <div className="space-y-0.5">
                                        <span className="text-xs font-bold text-foreground block">
                                            Automated Verification Certificate
                                        </span>
                                        <span className="text-[11px] text-muted-foreground block">
                                            Generate permanent verifiable certificate record for every completed scan.
                                        </span>
                                    </div>
                                    <input
                                        type="checkbox"
                                        checked={autoCertificate}
                                        onChange={(e) => setAutoCertificate(e.target.checked)}
                                        className="w-4 h-4 accent-foreground rounded cursor-pointer"
                                    />
                                </label>

                                <label className="flex items-center justify-between p-3 rounded-2xl bg-muted/30 border border-border/40 cursor-pointer hover:bg-muted/50 transition-colors">
                                    <div className="space-y-0.5">
                                        <span className="text-xs font-bold text-foreground block">
                                            Cryptographic SHA-256 Ledger
                                        </span>
                                        <span className="text-[11px] text-muted-foreground block">
                                            Calculate deterministic file digest before dispatching payload to inference cluster.
                                        </span>
                                    </div>
                                    <input
                                        type="checkbox"
                                        checked={hashLedger}
                                        onChange={(e) => setHashLedger(e.target.checked)}
                                        className="w-4 h-4 accent-foreground rounded cursor-pointer"
                                    />
                                </label>
                            </div>
                        </div>
                    </div>

                    {/* Section 2: Display & Appearance */}
                    <div className="p-6 rounded-2xl bg-card border border-border/80 shadow-sm space-y-6">
                        <div className="flex items-center gap-3">
                            <div className="p-2.5 rounded-xl bg-primary/10 text-primary border border-primary/20">
                                <Monitor className="w-5 h-5" />
                            </div>
                            <div>
                                <h2 className="text-base font-bold text-foreground font-sans">
                                    Interface & Display Preferences
                                </h2>
                                <p className="text-xs text-muted-foreground">
                                    Manage theme appearance and data grid layout density
                                </p>
                            </div>
                        </div>

                        <div className="grid sm:grid-cols-2 gap-4 pt-2">
                            {/* Theme Choice */}
                            <div className="p-4 rounded-xl bg-muted/30 border border-border/40 space-y-3">
                                <span className="text-xs font-bold text-foreground block">
                                    Color Theme
                                </span>
                                <div className="grid grid-cols-3 gap-2">
                                    <button
                                        type="button"
                                        onClick={() => setTheme("light")}
                                        className={`p-2.5 rounded-lg border text-xs font-semibold flex flex-col items-center gap-1.5 transition-colors ${
                                            theme === "light"
                                                ? "bg-foreground text-background border-foreground"
                                                : "bg-card text-foreground border-border hover:bg-muted"
                                        }`}
                                    >
                                        <Sun className="w-4 h-4" />
                                        <span>Light</span>
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setTheme("dark")}
                                        className={`p-2.5 rounded-lg border text-xs font-semibold flex flex-col items-center gap-1.5 transition-colors ${
                                            theme === "dark"
                                                ? "bg-foreground text-background border-foreground"
                                                : "bg-card text-foreground border-border hover:bg-muted"
                                        }`}
                                    >
                                        <Moon className="w-4 h-4" />
                                        <span>Dark</span>
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setTheme("system")}
                                        className={`p-2.5 rounded-lg border text-xs font-semibold flex flex-col items-center gap-1.5 transition-colors ${
                                            theme === "system"
                                                ? "bg-foreground text-background border-foreground"
                                                : "bg-card text-foreground border-border hover:bg-muted"
                                        }`}
                                    >
                                        <Monitor className="w-4 h-4" />
                                        <span>System</span>
                                    </button>
                                </div>
                            </div>

                            {/* Density Choice */}
                            <div className="p-4 rounded-xl bg-muted/30 border border-border/40 space-y-3">
                                <span className="text-xs font-bold text-foreground block">
                                    Ledger Row Density
                                </span>
                                <div className="grid grid-cols-2 gap-2">
                                    <button
                                        type="button"
                                        onClick={() => setTableDensity("comfortable")}
                                        className={`p-2.5 rounded-lg border text-xs font-semibold flex flex-col items-center gap-1.5 transition-colors ${
                                            tableDensity === "comfortable"
                                                ? "bg-foreground text-background border-foreground"
                                                : "bg-card text-foreground border-border hover:bg-muted"
                                        }`}
                                    >
                                        <span>Comfortable</span>
                                        <span className="text-[10px] opacity-75">56px row height</span>
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setTableDensity("compact")}
                                        className={`p-2.5 rounded-lg border text-xs font-semibold flex flex-col items-center gap-1.5 transition-colors ${
                                            tableDensity === "compact"
                                                ? "bg-foreground text-background border-foreground"
                                                : "bg-card text-foreground border-border hover:bg-muted"
                                        }`}
                                    >
                                        <span>Compact</span>
                                        <span className="text-[10px] opacity-75">42px row height</span>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right (4 cols): Cluster Health & Account */}
                <div className="lg:col-span-4 space-y-6">
                    {/* Cluster Diagnostics */}
                    <div className="p-5 rounded-2xl bg-card border border-border/80 shadow-sm space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5 font-mono">
                                <Cpu className="w-4 h-4 text-primary" />
                                <span>Cluster Status</span>
                            </span>
                            <span className="flex items-center gap-1 text-[11px] font-mono font-bold text-green-600">
                                <span className={`w-1.5 h-1.5 rounded-full ${backendStatus === "online" ? "bg-green-500 animate-pulse" : backendStatus === "checking" ? "bg-yellow-500 animate-pulse" : "bg-destructive"}`} />
                                {backendStatus === "online" ? "HEALTHY" : backendStatus.toUpperCase()}
                            </span>
                        </div>

                        <div className="space-y-3 text-xs">
                            <div className="p-3 rounded-xl bg-muted/30 border border-border/40 space-y-1">
                                <div className="flex justify-between font-semibold">
                                    <span className="text-foreground">Backend API Service</span>
                                    <span className={`font-mono ${backendStatus === "online" ? "text-green-600" : "text-destructive"}`}>
                                        {backendStatus === "online" ? "CONNECTED" : "OFFLINE"}
                                    </span>
                                </div>
                                <div className="flex justify-between text-[11px] text-muted-foreground">
                                    <span>Latency</span>
                                    <span className="font-mono">{backendPing !== null ? `${backendPing}ms` : "Unavailable"}</span>
                                </div>
                            </div>

                            <div className="p-3 rounded-xl bg-muted/30 border border-border/40 space-y-1">
                                <div className="flex justify-between font-semibold">
                                    <span className="text-foreground">ML Inference Engine</span>
                                    <span className="font-mono text-foreground">
                                        {healthDetails?.mlService || (backendStatus === "online" ? "AVAILABLE" : "UNAVAILABLE")}
                                    </span>
                                </div>
                                <div className="flex justify-between text-[11px] text-muted-foreground">
                                    <span>Pipeline</span>
                                    <span className="font-mono">SigLIP ViT + ELA</span>
                                </div>
                            </div>

                            <div className="p-3 rounded-xl bg-muted/30 border border-border/40 space-y-1">
                                <div className="flex justify-between font-semibold">
                                    <span className="text-foreground">Database Connectivity</span>
                                    <span className={`font-mono ${healthDetails?.database === "UP" || backendStatus === "online" ? "text-green-600" : "text-destructive"}`}>
                                        {healthDetails?.database || (backendStatus === "online" ? "CONNECTED" : "DISCONNECTED")}
                                    </span>
                                </div>
                                <div className="flex justify-between text-[11px] text-muted-foreground">
                                    <span>Ledger Storage</span>
                                    <span className="font-mono">Live PostgreSQL</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Account & Session Card */}
                    <div className="p-5 rounded-2xl bg-card border border-border/80 shadow-sm space-y-4">
                        <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                            <User className="w-4 h-4 text-primary" />
                            <span>Operator Account</span>
                        </span>

                        {userLoaded && isSignedIn && user ? (
                            <div className="space-y-3">
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-full bg-foreground text-background flex items-center justify-center font-bold text-sm">
                                        {user.firstName ? user.firstName[0] : "U"}
                                    </div>
                                    <div className="min-w-0">
                                        <p className="text-xs font-bold text-foreground truncate">
                                            {user.fullName || user.username || "Verified Analyst"}
                                        </p>
                                        <p className="text-[11px] text-muted-foreground truncate">
                                            {user.primaryEmailAddress?.emailAddress || "Analyst Session"}
                                        </p>
                                    </div>
                                </div>
                                <div className="p-2.5 rounded-xl bg-muted/40 border border-border/50 text-[11px] text-muted-foreground space-y-1">
                                    <div className="flex justify-between">
                                        <span>Role</span>
                                        <span className="font-bold text-foreground">Forensics Analyst</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Auth Provider</span>
                                        <span className="font-mono">Clerk Enterprise</span>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="space-y-3 text-xs">
                                <p className="text-muted-foreground">
                                    Currently operating in sovereign guest mode. Sign in to synchronize scans across devices.
                                </p>
                                <Link
                                    href="/sign-in"
                                    className="w-full py-2.5 rounded-xl bg-foreground text-background text-xs font-semibold hover:bg-foreground/90 transition-colors flex items-center justify-center gap-1.5"
                                >
                                    <span>Sign In with Clerk</span>
                                    <ArrowUpRight className="w-3.5 h-3.5" />
                                </Link>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
