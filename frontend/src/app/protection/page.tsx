"use client";

import { UploadDropzone } from "@/components/shared/UploadDropzone";
import { PlatformReportCard } from "@/components/shared/PlatformReportCard";
import { useState } from "react";
import { SearchCode, AlertTriangle, Fingerprint } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";

export default function ProtectionPage() {
    const [file, setFile] = useState<File | null>(null);
    const [imageUrl, setImageUrl] = useState<string>("");
    const [scanning, setScanning] = useState(false);
    const [results, setResults] = useState(false);

    const handleUpload = (file: File) => {
        setFile(file);
        setImageUrl(URL.createObjectURL(file));
        setScanning(true);
        setResults(false);

        // Simulate dataset matching
        setTimeout(() => {
            setScanning(false);
            setResults(true);
        }, 4000);
    };

    return (
        <div className="min-h-[calc(100vh-4rem)] bg-background">
            <div className="max-w-4xl mx-auto px-4 py-12 md:py-24 space-y-12">
                <div className="text-center space-y-4">
                    <Fingerprint className="w-16 h-16 text-primary mx-auto mb-6 bg-primary/10 p-4 rounded-3xl" />
                    <h1 className="text-4xl font-bold tracking-tight text-balance">Digital Identity Protection</h1>
                    <p className="text-muted-foreground text-lg max-w-2xl mx-auto text-balance">
                        Upload your verified photo. We scan the web and hidden datasets to find unauthorized AI-generated images using your likeness.
                    </p>
                </div>

                {!file && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="bg-card p-4 sm:p-8 border shadow-sm rounded-[2rem] max-w-2xl mx-auto"
                    >
                        <UploadDropzone onUpload={handleUpload} />
                    </motion.div>
                )}

                <AnimatePresence>
                    {scanning && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.9 }}
                            className="flex flex-col items-center justify-center p-12 bg-card border rounded-[2rem] shadow-sm max-w-xl mx-auto text-center"
                        >
                            <div className="relative mb-8">
                                <div className="w-32 h-32 rounded-full overflow-hidden border-4 border-primary/20 p-2 border-dashed animate-spin-slow">
                                    <div className="w-full h-full rounded-full overflow-hidden relative border-2 border-solid border-primary/50 animate-pulse">
                                        <Image src={imageUrl} alt="Uploaded face" fill className="object-cover" />
                                        <div className="absolute inset-0 bg-primary/20 mix-blend-overlay"></div>
                                    </div>
                                </div>
                                <SearchCode className="absolute -bottom-4 -right-4 w-12 h-12 text-primary p-2 bg-background border rounded-full shadow-lg" />
                            </div>
                            <h3 className="text-2xl font-bold mb-2">Scanning Web Datasets...</h3>
                            <p className="text-muted-foreground text-balance">Matching your facial embeddings against known deepfake generation platforms and adult forums.</p>
                        </motion.div>
                    )}
                </AnimatePresence>

                <AnimatePresence>
                    {results && (
                        <motion.div
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="space-y-8 max-w-3xl mx-auto"
                        >
                            <div className="bg-destructive/10 border-destructive/20 text-destructive border p-6 rounded-[2rem] flex items-start gap-4 shadow-sm shadow-destructive/5">
                                <AlertTriangle className="w-8 h-8 shrink-0 mt-1" />
                                <div>
                                    <h3 className="text-xl font-bold mb-2 text-foreground">Possible identity misuse detected</h3>
                                    <p className="text-muted-foreground text-balance">
                                        We found 3 simulated high-confidence matches of unauthorized synthetic media using your likeness across public platforms.
                                    </p>
                                </div>
                            </div>

                            <div className="space-y-6">
                                <div className="flex items-center justify-between border-b pb-4">
                                    <h4 className="font-semibold text-lg flex items-center gap-2"><SearchCode className="w-5 h-5 text-primary" /> Active Matches</h4>
                                    <span className="text-sm font-medium text-muted-foreground px-3 py-1 bg-muted rounded-full">3 Sources found</span>
                                </div>

                                <div className="flex flex-col space-y-4">
                                    <PlatformReportCard
                                        platform="Reddit"
                                        url="reddit.com/r/deepfake_sim/post_1029"
                                        matchConfidence={98}
                                        dateDetected={new Date().toLocaleDateString()}
                                    />
                                    <PlatformReportCard
                                        platform="Patreon (Creator: AI_Gen_Artist)"
                                        url="patreon.com/posts/ai-exclusive-pack"
                                        matchConfidence={92}
                                        dateDetected="2 Days Ago"
                                    />
                                    <PlatformReportCard
                                        platform="Simulated Adult Content Forum"
                                        url="fake-forum.onion/thread/99283"
                                        matchConfidence={88}
                                        dateDetected="1 Week Ago"
                                    />
                                </div>
                            </div>

                            <div className="flex justify-center pt-8">
                                <button
                                    onClick={() => { setFile(null); setResults(false); }}
                                    className="px-6 py-3 font-semibold text-muted-foreground hover:text-foreground hover:bg-muted rounded-xl transition-all"
                                >
                                    Scan Another Profile
                                </button>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
