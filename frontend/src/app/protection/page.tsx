"use client";

import { UploadDropzone } from "@/components/shared/UploadDropzone";
import { PlatformReportCard } from "@/components/shared/PlatformReportCard";
import { useState, useEffect } from "react";
import {
    SearchCode,
    AlertTriangle,
    Fingerprint,
    Radio,
    RefreshCw,
    FileText,
    Copy,
    Check,
    Download,
    Lock,
    ShieldCheck,
    AlertCircle
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner";
import { useAuth, SignInButton } from "@clerk/nextjs";
import { detectionApi, IdentityProtectionResult } from "@/lib/api";
import { buildTruthfulDmcaNotice } from "@/lib/dmcaNotice";

const scanMilestones = [
    "Hashing facial landmarks and feature embeddings...",
    "Scanning synthetic training pools and web indices...",
    "Cross-referencing face-swap repositories and imageboards...",
    "Aggregating infringement risk scores...",
];

export default function ProtectionPage() {
    const { isSignedIn, isLoaded } = useAuth();
    const [file, setFile] = useState<File | null>(null);
    const [imageUrl, setImageUrl] = useState<string>("");
    const [scanning, setScanning] = useState(false);
    const [scanError, setScanError] = useState<string | null>(null);
    const [activeMilestone, setActiveMilestone] = useState(0);
    const [identityData, setIdentityData] = useState<IdentityProtectionResult | null>(null);
    const [copiedDmca, setCopiedDmca] = useState(false);

    useEffect(() => {
        if (!scanning) return;
        const interval = setInterval(() => {
            setActiveMilestone((prev) => (prev + 1) % scanMilestones.length);
        }, 1100);
        return () => clearInterval(interval);
    }, [scanning]);

    // Clean up temporary image URL on unmount
    useEffect(() => {
        return () => {
            if (imageUrl) {
                URL.revokeObjectURL(imageUrl);
            }
        };
    }, [imageUrl]);

    const handleUpload = async (uploadedFile: File) => {
        if (imageUrl) {
            URL.revokeObjectURL(imageUrl);
        }
        setFile(uploadedFile);
        setImageUrl(URL.createObjectURL(uploadedFile));
        setScanning(true);
        setScanError(null);
        setIdentityData(null);
        setActiveMilestone(0);

        try {
            const data = await detectionApi.identityProtect(uploadedFile);
            setIdentityData(data);
            setScanning(false);

            const matchesFound = data.total_infringements_found || data.detected_matches?.length || 0;
            if (matchesFound > 0) {
                toast.warning(`Identity scan complete: ${matchesFound} potential likeness match(es) detected.`);
            } else {
                toast.success("Identity scan complete: No unauthorized likeness detected.");
            }
        } catch (err: unknown) {
            console.error("Identity protection scan error:", err);
            setScanning(false);
            const message = err instanceof Error ? err.message : "Service temporarily unavailable. Please retry.";
            setScanError(message);
            toast.error(`Scan failed: ${message}`);
        }
    };

    const handleReset = () => {
        setFile(null);
        setScanning(false);
        setScanError(null);
        setIdentityData(null);
        if (imageUrl) {
            URL.revokeObjectURL(imageUrl);
            setImageUrl("");
        }
    };

    const detectedMatches = identityData?.detected_matches ?? [];
    const totalMatches = identityData?.total_infringements_found ?? detectedMatches.length;
    const hasFindings = totalMatches > 0;

    // Generate truthful DMCA notice only when actual matches exist
    const generateDmcaNotice = (): string => {
        if (!hasFindings || detectedMatches.length === 0) {
            return "No verified infringing URLs detected. DMCA notice generation is disabled when 0 infringements are identified.";
        }
        return buildTruthfulDmcaNotice({
            matches: detectedMatches,
            biometricHash: identityData?.biometric_hash,
        }) || "No verified infringing URLs detected. DMCA notice generation is disabled when 0 infringements are identified.";
    };

    const handleCopyDmca = () => {
        if (!hasFindings) {
            toast.error("Cannot generate DMCA notice: No infringing URLs were found.");
            return;
        }
        const text = identityData?.dmca_notice_template || generateDmcaNotice();
        navigator.clipboard.writeText(text);
        setCopiedDmca(true);
        toast.success("Legal DMCA Takedown Notice copied to clipboard");
        setTimeout(() => setCopiedDmca(false), 2000);
    };

    const handleDownloadDmca = () => {
        if (!hasFindings) {
            toast.error("Cannot download DMCA notice: No infringing URLs were found.");
            return;
        }
        const text = identityData?.dmca_notice_template || generateDmcaNotice();
        const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `DMCA-Takedown-Notice-${Date.now()}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toast.success("DMCA legal notice letter downloaded");
    };

    const authRequired = isLoaded && !isSignedIn;

    return (
        <div className="min-h-[calc(100vh-4rem)] bg-background">
            <div className="max-w-4xl mx-auto px-4 py-8 md:py-16 space-y-12">
                {/* Hero Header */}
                <div className="text-center space-y-4">
                    <div className="inline-flex items-center justify-center p-4 bg-primary/10 rounded-2xl mb-2 text-primary border border-primary/20">
                        <Fingerprint className="w-10 h-10" />
                    </div>
                    <h1 className="text-4xl md:text-5xl font-black tracking-tight text-foreground text-balance">
                        Digital Likeness Protection
                    </h1>
                    <p className="text-muted-foreground text-base md:text-lg max-w-2xl mx-auto text-balance">
                        Protect your personal biometric identity. Search open web mirrors, AI generator forums, and indexed databases for unauthorized synthetic face clones.
                    </p>
                </div>

                {/* Authentication Requirement Prompt */}
                {authRequired && (
                    <motion.div
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-card border-2 border-primary/30 p-8 rounded-2xl text-center space-y-5 shadow-lg max-w-xl mx-auto"
                    >
                        <div className="w-12 h-12 rounded-xl bg-primary/10 text-primary flex items-center justify-center mx-auto border border-primary/20">
                            <Lock className="w-6 h-6" />
                        </div>
                        <div className="space-y-1.5">
                            <h3 className="text-xl font-bold">Authentication Required</h3>
                            <p className="text-xs text-muted-foreground leading-relaxed">
                                Digital likeness surveillance searches require an authenticated account to prevent unauthorized facial indexing and comply with privacy regulations.
                            </p>
                        </div>
                        <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
                            <SignInButton mode="modal">
                                <button className="w-full sm:w-auto px-6 py-2.5 rounded-xl bg-foreground text-background text-xs font-bold uppercase tracking-wider hover:bg-foreground/90 transition-all shadow-sm">
                                    Sign In with Clerk
                                </button>
                            </SignInButton>
                        </div>
                    </motion.div>
                )}

                {/* Upload Section */}
                {!authRequired && !file && !scanning && !identityData && !scanError && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.96 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="bg-card p-4 sm:p-8 border border-border shadow-sm rounded-2xl max-w-2xl mx-auto space-y-6"
                    >
                        <div className="text-center space-y-1">
                            <h2 className="text-lg font-bold">Provide Reference Portrait</h2>
                            <p className="text-xs text-muted-foreground">
                                Upload a front-facing image. High resolution provides the most accurate feature matching.
                            </p>
                        </div>
                        <UploadDropzone onUpload={handleUpload} />
                    </motion.div>
                )}

                {/* Scanning Radar State */}
                <AnimatePresence>
                    {scanning && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            className="flex flex-col items-center justify-center p-8 sm:p-12 bg-card border border-border rounded-2xl shadow-sm max-w-xl mx-auto text-center space-y-6"
                        >
                            <div className="relative mb-2">
                                <div className="w-36 h-36 rounded-full border-4 border-dashed border-primary/30 p-2 animate-spin-slow">
                                    <div className="w-full h-full rounded-full overflow-hidden relative border-2 border-primary/60">
                                        {/* eslint-disable-next-line @next/next/no-img-element */}
                                        <img
                                            src={imageUrl}
                                            alt="Reference face"
                                            className="w-full h-full object-cover"
                                        />
                                        <div className="absolute inset-0 bg-primary/20 mix-blend-overlay animate-pulse" />
                                    </div>
                                </div>
                                <div className="absolute -bottom-2 -right-2 w-10 h-10 rounded-full bg-background border border-border flex items-center justify-center shadow-lg text-primary">
                                    <Radio className="w-5 h-5 animate-pulse" />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <h3 className="text-2xl font-bold tracking-tight">Scanning Global Vector Index...</h3>
                                <p className="text-sm font-medium text-primary">
                                    {scanMilestones[activeMilestone]}
                                </p>
                                <p className="text-xs text-muted-foreground max-w-md mx-auto">
                                    Comparing localized facial embeddings against indexed generative media repositories.
                                </p>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Error State */}
                <AnimatePresence>
                    {scanError && !scanning && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="bg-destructive/10 border border-destructive/20 text-destructive p-8 rounded-2xl max-w-xl mx-auto text-center space-y-4 shadow-sm"
                        >
                            <AlertCircle className="w-10 h-10 mx-auto text-destructive" />
                            <div className="space-y-1">
                                <h3 className="text-lg font-bold text-foreground">Analysis Request Failed</h3>
                                <p className="text-xs text-muted-foreground max-w-md mx-auto">
                                    {scanError}
                                </p>
                            </div>
                            <div className="pt-2">
                                <button
                                    onClick={handleReset}
                                    className="px-6 py-2.5 rounded-xl bg-foreground text-background text-xs font-bold uppercase tracking-wider hover:bg-foreground/90 transition-all shadow-sm inline-flex items-center gap-2"
                                >
                                    <RefreshCw className="w-3.5 h-3.5" />
                                    Retry with Another Image
                                </button>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Results View - Clean / No Findings */}
                <AnimatePresence>
                    {!scanning && identityData && !hasFindings && (
                        <motion.div
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="space-y-8 max-w-3xl mx-auto"
                        >
                            <div className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 p-6 rounded-2xl flex items-start gap-4 shadow-sm">
                                <ShieldCheck className="w-7 h-7 shrink-0 mt-1" />
                                <div className="space-y-1">
                                    <h3 className="text-lg font-bold text-foreground">
                                        No Unauthorized Synthetic Likeness Detected
                                    </h3>
                                    <p className="text-sm text-muted-foreground text-balance">
                                        Vector surveillance completed. Zero unauthorized synthetic media matching this facial embedding were identified across indexed sources.
                                    </p>
                                </div>
                            </div>

                            <div className="bg-card border border-border p-6 rounded-2xl space-y-4">
                                <div className="flex items-center justify-between border-b border-border pb-3">
                                    <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Scan Status</span>
                                    <span className="text-xs px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold">
                                        Nominal / Clear
                                    </span>
                                </div>
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
                                    <div>
                                        <p className="text-muted-foreground">Biometric SHA-256 Hash</p>
                                        <p className="font-mono text-foreground mt-0.5 truncate">{identityData.biometric_hash || "Unavailable"}</p>
                                    </div>
                                    <div>
                                        <p className="text-muted-foreground">Indexed Matches</p>
                                        <p className="font-semibold text-foreground mt-0.5">0 records</p>
                                    </div>
                                </div>
                            </div>

                            <div className="flex justify-center pt-4">
                                <button
                                    type="button"
                                    onClick={handleReset}
                                    className="px-6 py-3 font-semibold text-foreground bg-card border border-border hover:bg-muted rounded-xl transition-all flex items-center justify-center gap-2 text-sm shadow-sm"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                    Scan Another Portrait
                                </button>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Results View - With Genuine Findings */}
                <AnimatePresence>
                    {!scanning && identityData && hasFindings && (
                        <motion.div
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="space-y-8 max-w-3xl mx-auto"
                        >
                            {/* Threat Banner */}
                            <div className="bg-destructive/10 border border-destructive/20 text-destructive p-6 rounded-2xl flex items-start gap-4 shadow-sm">
                                <AlertTriangle className="w-7 h-7 shrink-0 mt-1" />
                                <div className="space-y-1">
                                    <h3 className="text-lg font-bold text-foreground">
                                        Potential Visual Likeness Matches Identified (Threat Level: {identityData.threat_level || "ELEVATED"})
                                    </h3>
                                    <p className="text-sm text-muted-foreground text-balance">
                                        Identified {totalMatches} potential visual match(es) sharing facial embedding features with your reference portrait. Note: Potential visual similarity does not independently prove unauthorized misuse or identity theft.
                                    </p>
                                </div>
                            </div>

                            {/* Threat List */}
                            <div className="space-y-5">
                                <div className="flex items-center justify-between border-b border-border pb-4">
                                    <div className="flex items-center gap-2">
                                        <SearchCode className="w-5 h-5 text-primary" />
                                        <h4 className="font-bold text-lg text-foreground">Identified Matches</h4>
                                    </div>
                                    <span className="text-xs font-semibold px-3 py-1 bg-muted rounded-full text-foreground border border-border">
                                        {totalMatches} Match{totalMatches > 1 ? "es" : ""} Found
                                    </span>
                                </div>

                                <div className="flex flex-col space-y-3.5">
                                    {detectedMatches.map((m, idx) => (
                                        <PlatformReportCard
                                            key={idx}
                                            platform={m.platform}
                                            url={m.url}
                                            matchConfidence={m.similarity_score}
                                            dateDetected={m.date_spotted}
                                        />
                                    ))}
                                </div>
                            </div>

                            {/* DMCA Legal Takedown Notice Generator Tool */}
                            <div className="p-6 rounded-2xl bg-card border border-border shadow-sm space-y-4">
                                <div className="flex items-center justify-between">
                                    <div className="space-y-0.5">
                                        <h4 className="font-bold text-base flex items-center gap-2">
                                            <FileText className="w-4 h-4 text-primary" />
                                            Automated DMCA Takedown Notice Letter
                                        </h4>
                                        <p className="text-xs text-muted-foreground">
                                            Populated with verified detected URLs and your cryptographic image SHA-256 hash.
                                        </p>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={handleCopyDmca}
                                            className="px-3.5 py-1.5 rounded-xl border border-border text-xs font-semibold hover:bg-muted flex items-center gap-1.5"
                                        >
                                            {copiedDmca ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                                            <span>{copiedDmca ? "Copied" : "Copy"}</span>
                                        </button>
                                        <button
                                            onClick={handleDownloadDmca}
                                            className="px-3.5 py-1.5 rounded-xl bg-foreground text-background text-xs font-bold hover:bg-foreground/90 flex items-center gap-1.5 shadow-sm"
                                        >
                                            <Download className="w-3.5 h-3.5" />
                                            <span>Download .txt</span>
                                        </button>
                                    </div>
                                </div>

                                <pre className="font-mono text-xs text-muted-foreground p-4 rounded-2xl bg-muted/50 border border-border/60 max-h-48 overflow-y-auto whitespace-pre-wrap leading-relaxed">
                                    {identityData.dmca_notice_template || generateDmcaNotice()}
                                </pre>
                            </div>

                            {/* Footer Actions */}
                            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-6 border-t border-border">
                                <button
                                    type="button"
                                    onClick={handleReset}
                                    className="w-full sm:w-auto px-6 py-3 font-semibold text-foreground bg-card border border-border hover:bg-muted rounded-xl transition-all flex items-center justify-center gap-2 text-sm shadow-sm"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                    Scan Another Portrait
                                </button>
                                <button
                                    type="button"
                                    onClick={handleDownloadDmca}
                                    className="w-full sm:w-auto px-6 py-3 font-semibold text-primary-foreground bg-primary hover:bg-primary/90 rounded-xl transition-all flex items-center justify-center gap-2 text-sm shadow-sm"
                                >
                                    <Download className="w-4 h-4" />
                                    Download Full DMCA Docket
                                </button>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
