"use client";

import { UploadDropzone } from "@/components/shared/UploadDropzone";
import { useState, useEffect } from "react";
import { SearchCode, Fingerprint, ExternalLink, Globe, RotateCcw, Loader2, ShieldCheck, AlertTriangle, Eye, Instagram, Twitter, Facebook, Linkedin } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import axios from "axios";

const ML_SERVICE_URL = process.env.NEXT_PUBLIC_ML_URL || "http://localhost:8000";

interface VisualMatch {
    position: number;
    title: string;
    link: string;
    source: string;
    source_icon: string;
    thumbnail: string;
}

interface SearchResults {
    success: boolean;
    image_url: string;
    total_matches: number;
    visual_matches: VisualMatch[];
    search_metadata: {
        google_lens_url: string;
    };
}

export default function ProtectionPage() {
    const [file, setFile] = useState<File | null>(null);
    const [imageUrl, setImageUrl] = useState<string>("");
    const [scanning, setScanning] = useState(false);
    const [results, setResults] = useState<SearchResults | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [progress, setProgress] = useState("");

    // Dynamic scanning text
    useEffect(() => {
        if (!scanning) return;
        const texts = [
            "Scanning Google Images...",
            "Checking Instagram & Facebook...",
            "Searching Twitter databases...",
            "Analyzing public web archives...",
            "Correlating visual matches...",
        ];
        let i = 0;
        setProgress(texts[0]);
        const interval = setInterval(() => {
            i = (i + 1) % texts.length;
            setProgress(texts[i]);
        }, 2500);
        return () => clearInterval(interval);
    }, [scanning]);

    const handleUpload = async (uploadedFile: File) => {
        setFile(uploadedFile);
        setImageUrl(URL.createObjectURL(uploadedFile));
        setScanning(true);
        setResults(null);
        setError(null);

        try {
            setProgress("Uploading image...");
            const formData = new FormData();
            formData.append("file", uploadedFile);
            const minDelay = new Promise(resolve => setTimeout(resolve, 10000)); // Ensure at least 10s animation
            
            const [response] = await Promise.all([
                axios.post<SearchResults>(
                    `${ML_SERVICE_URL}/reverse-search`,
                    formData,
                    {
                        headers: { "Content-Type": "multipart/form-data" },
                        timeout: 60000,
                    }
                ),
                minDelay
            ]);

            setResults(response.data);
        } catch (err: unknown) {
            console.error("Reverse search failed:", err);
            if (axios.isAxiosError(err) && err.response?.status === 503) {
                setError("SerpAPI key not configured. Set SERP_API_KEY environment variable on the ML service.");
            } else {
                setError("Failed to perform reverse image search. Please check that the ML service is running.");
            }
        } finally {
            setScanning(false);
            setProgress("");
        }
    };

    const handleReset = () => {
        if (imageUrl) URL.revokeObjectURL(imageUrl);
        setFile(null);
        setImageUrl("");
        setScanning(false);
        setResults(null);
        setError(null);
    };

    const getDomain = (url: string) => {
        try {
            return new URL(url).hostname.replace("www.", "");
        } catch {
            return url;
        }
    };

    return (
        <div className="min-h-[calc(100vh-4rem)] bg-background">
            <div className="max-w-5xl mx-auto px-4 py-12 md:py-20 space-y-10">
                {/* Header */}
                <div className="text-center space-y-4">
                    <Fingerprint className="w-16 h-16 text-accent mx-auto mb-6 bg-accent/10 p-4 rounded-3xl" />
                    <h1 className="text-4xl font-bold tracking-tight text-balance">Digital Identity Protection</h1>
                    <p className="text-muted-foreground text-lg max-w-2xl mx-auto text-balance">
                        Upload your photo and we&apos;ll scan the web using Google Lens to find where your image appears online — exact matches, similar images, and potential misuse.
                    </p>
                </div>

                {/* Upload Area */}
                {!file && !scanning && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="bg-card p-4 sm:p-8 border shadow-sm rounded-[2rem] max-w-2xl mx-auto"
                    >
                        <UploadDropzone onUpload={handleUpload} />
                    </motion.div>
                )}

                {/* Scanning State */}
                <AnimatePresence>
                    {scanning && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.9 }}
                            className="flex flex-col items-center justify-center p-12 bg-card border rounded-[2rem] shadow-sm max-w-xl mx-auto text-center"
                        >
                            <div className="relative mb-8 w-64 h-64 flex items-center justify-center">
                                {/* Sonar Pulse Glow */}
                                <div className="absolute inset-0 bg-accent/10 rounded-full blur-2xl animate-pulse" />
                                
                                {/* Orbit Ring 1 (Outer) */}
                                <motion.div 
                                    animate={{ rotate: 360 }} 
                                    transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
                                    className="absolute inset-0 rounded-full border border-dashed border-accent/30"
                                >
                                    <div className="absolute -top-4 left-1/2 -translate-x-1/2 w-8 h-8 bg-card border rounded-full flex items-center justify-center text-accent shadow-sm"><Instagram className="w-4 h-4" /></div>
                                    <div className="absolute -bottom-4 left-1/2 -translate-x-1/2 w-8 h-8 bg-card border rounded-full flex items-center justify-center text-accent shadow-sm"><Twitter className="w-4 h-4" /></div>
                                </motion.div>

                                {/* Orbit Ring 2 (Inner) */}
                                <motion.div 
                                    animate={{ rotate: -360 }} 
                                    transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                                    className="absolute inset-8 rounded-full border border-dashed border-accent/40"
                                >
                                    <div className="absolute top-1/2 -left-4 -translate-y-1/2 w-8 h-8 bg-card border rounded-full flex items-center justify-center text-accent shadow-sm"><Facebook className="w-4 h-4" /></div>
                                    <div className="absolute top-1/2 -right-4 -translate-y-1/2 w-8 h-8 bg-card border rounded-full flex items-center justify-center text-accent shadow-sm"><Linkedin className="w-4 h-4" /></div>
                                </motion.div>

                                {/* Central Image */}
                                <div className="w-32 h-32 rounded-full overflow-hidden relative z-10 border-4 border-background shadow-xl">
                                    <Image src={imageUrl} alt="Scanning" fill className="object-cover" />
                                    <div className="absolute inset-0 bg-accent/20 mix-blend-overlay animate-pulse"></div>
                                </div>
                                
                                {/* Sweeping Radar Beam */}
                                <motion.div 
                                    animate={{ rotate: 360 }}
                                    transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                                    className="absolute inset-0 z-20 rounded-full overflow-hidden pointer-events-none"
                                >
                                    <div className="w-1/2 h-1/2 bg-gradient-to-br from-accent/30 to-transparent origin-bottom-right" />
                                </motion.div>
                            </div>
                            <h3 className="text-2xl font-bold mb-2">Scanning the Web...</h3>
                            <p className="text-muted-foreground text-balance h-6 font-medium">{progress}</p>
                            <div className="flex items-center gap-2 mt-6 text-accent text-sm font-medium">
                                <Loader2 className="w-4 h-4 animate-spin" />
                                This may take 10-20 seconds
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Error State */}
                {error && !scanning && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="space-y-6 max-w-2xl mx-auto"
                    >
                        <div className="bg-destructive/8 border-destructive/20 border p-6 rounded-2xl flex items-start gap-4">
                            <AlertTriangle className="w-6 h-6 text-destructive shrink-0 mt-0.5" />
                            <div>
                                <h3 className="text-[15px] font-semibold text-foreground mb-1">Search Failed</h3>
                                <p className="text-[13px] text-muted-foreground">{error}</p>
                            </div>
                        </div>
                        <div className="flex justify-center">
                            <button onClick={handleReset}
                                className="flex items-center gap-2 px-5 py-2.5 text-[14px] font-medium text-muted-foreground hover:text-foreground hover:bg-muted rounded-xl transition-all">
                                <RotateCcw className="w-4 h-4" /> Try Another Image
                            </button>
                        </div>
                    </motion.div>
                )}

                {/* Results */}
                <AnimatePresence>
                    {results && !scanning && (
                        <motion.div
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="space-y-6"
                        >
                            {/* Result Summary */}
                            <div className={`border p-6 rounded-2xl flex items-start gap-4 shadow-sm ${
                                results.total_matches > 0
                                    ? "bg-orange-500/8 border-orange-500/20"
                                    : "bg-green-500/8 border-green-500/20"
                            }`}>
                                {results.total_matches > 0 ? (
                                    <Eye className="w-7 h-7 text-orange-500 shrink-0 mt-0.5" />
                                ) : (
                                    <ShieldCheck className="w-7 h-7 text-green-500 shrink-0 mt-0.5" />
                                )}
                                <div>
                                    <h3 className="text-lg font-bold text-foreground mb-1">
                                        {results.total_matches > 0
                                            ? `Found ${results.total_matches} visual match${results.total_matches !== 1 ? "es" : ""} across the web`
                                            : "No matches found"
                                        }
                                    </h3>
                                    <p className="text-[14px] text-muted-foreground">
                                        {results.total_matches > 0
                                            ? "Your image was found on the following websites. Review each match to check for unauthorized use."
                                            : "Your image does not appear to be widely distributed online. No exact or similar matches were found."
                                        }
                                    </p>
                                </div>
                            </div>

                            {/* Uploaded Image Preview */}
                            <div className="flex items-center gap-4 p-4 bg-card border rounded-xl" style={{ borderColor: "hsl(var(--border) / 0.4)" }}>
                                <div className="w-16 h-16 rounded-lg overflow-hidden border relative shrink-0">
                                    <Image src={imageUrl} alt="Uploaded" fill className="object-cover" />
                                </div>
                                <div className="min-w-0 flex-1">
                                    <p className="text-[14px] font-semibold text-foreground truncate">{file?.name}</p>
                                    <p className="text-[12px] text-muted-foreground">
                                        {file ? `${(file.size / 1024).toFixed(1)} KB` : ""} • {results.total_matches} matches found
                                    </p>
                                </div>
                                {results.search_metadata?.google_lens_url && (
                                    <a href={results.search_metadata.google_lens_url} target="_blank" rel="noopener noreferrer"
                                        className="flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium text-accent hover:text-accent/80 transition-colors shrink-0">
                                        <Globe className="w-3.5 h-3.5" /> View on Google Lens <ExternalLink className="w-3 h-3" />
                                    </a>
                                )}
                            </div>

                            {/* Visual Matches Grid */}
                            {results.visual_matches.length > 0 && (
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <h4 className="text-[15px] font-semibold text-foreground flex items-center gap-2">
                                            <SearchCode className="w-4 h-4 text-accent" /> Visual Matches
                                        </h4>
                                        <span className="text-[12px] font-medium text-muted-foreground px-3 py-1 bg-muted rounded-full">
                                            {results.visual_matches.length} sources
                                        </span>
                                    </div>

                                    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                                        {results.visual_matches.map((match, i) => (
                                            <motion.a
                                                key={i}
                                                href={match.link}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                initial={{ opacity: 0, y: 10 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                transition={{ delay: i * 0.05 }}
                                                className="group bg-card border rounded-xl overflow-hidden hover:shadow-lg hover:border-accent/30 transition-all duration-200"
                                                style={{ borderColor: "hsl(var(--border) / 0.5)" }}
                                            >
                                                {/* Thumbnail */}
                                                <div className="aspect-[4/3] relative bg-muted/30 overflow-hidden">
                                                    {match.thumbnail ? (
                                                        <img
                                                            src={match.thumbnail}
                                                            alt={match.title}
                                                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                                                            loading="lazy"
                                                        />
                                                    ) : (
                                                        <div className="w-full h-full flex items-center justify-center text-muted-foreground">
                                                            <Globe className="w-8 h-8" />
                                                        </div>
                                                    )}
                                                    {/* Hover overlay */}
                                                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors duration-200 flex items-center justify-center">
                                                        <ExternalLink className="w-5 h-5 text-white opacity-0 group-hover:opacity-100 transition-opacity duration-200 drop-shadow-lg" />
                                                    </div>
                                                </div>

                                                {/* Info */}
                                                <div className="p-3.5 space-y-1.5">
                                                    <p className="text-[13px] font-semibold text-foreground line-clamp-2 leading-snug group-hover:text-accent transition-colors">
                                                        {match.title || "Untitled"}
                                                    </p>
                                                    <div className="flex items-center gap-2">
                                                        {match.source_icon && (
                                                            <img src={match.source_icon} alt="" className="w-4 h-4 rounded-sm" />
                                                        )}
                                                        <span className="text-[12px] text-muted-foreground truncate">
                                                            {match.source || getDomain(match.link)}
                                                        </span>
                                                    </div>
                                                </div>
                                            </motion.a>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Actions */}
                            <div className="flex flex-col sm:flex-row justify-center gap-3 pt-4">
                                <button onClick={handleReset}
                                    className="flex items-center justify-center gap-2 px-6 py-3 font-semibold text-muted-foreground hover:text-foreground hover:bg-muted rounded-xl transition-all text-[14px]">
                                    <RotateCcw className="w-4 h-4" /> Scan Another Image
                                </button>
                                {results.search_metadata?.google_lens_url && (
                                    <a href={results.search_metadata.google_lens_url} target="_blank" rel="noopener noreferrer"
                                        className="flex items-center justify-center gap-2 px-6 py-3 bg-foreground text-background font-semibold rounded-xl hover:opacity-90 transition-all text-[14px]">
                                        <Globe className="w-4 h-4" /> View Full Results on Google
                                        <ExternalLink className="w-3.5 h-3.5" />
                                    </a>
                                )}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
