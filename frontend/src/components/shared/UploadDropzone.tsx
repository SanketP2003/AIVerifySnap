"use client";

import { useCallback, useState } from "react";
import { useDropzone, FileRejection } from "react-dropzone";
import { UploadCloud, Image as ImageIcon, FileImage, X, AlertCircle, FileCheck, FolderOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";

interface UploadDropzoneProps {
    onUpload: (file: File) => void;
    isUploading?: boolean;
}

function formatFileSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function UploadDropzone({ onUpload, isUploading = false }: UploadDropzoneProps) {
    const [error, setError] = useState<string | null>(null);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);

    const clearSelection = useCallback(() => {
        setSelectedFile(null);
        setError(null);
        setDimensions(null);
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            setPreviewUrl(null);
        }
    }, [previewUrl]);

    const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
        if (rejectedFiles.length > 0) {
            const rejection = rejectedFiles[0];
            if (rejection.errors.some(e => e.code === "file-too-large")) {
                setError("File is too large. Maximum size is 10 MB.");
            } else if (rejection.errors.some(e => e.code === "file-invalid-type")) {
                setError("Invalid file type. Please upload a JPEG, PNG, or WebP image.");
            } else {
                setError("Please upload a valid image file (JPEG, PNG, or WebP).");
            }
            return;
        }
        setError(null);
        if (acceptedFiles.length > 0) {
            const file = acceptedFiles[0];
            setSelectedFile(file);
            const url = URL.createObjectURL(file);
            setPreviewUrl(url);

            // Read dimensions
            const img = new window.Image();
            img.onload = () => {
                setDimensions({ width: img.naturalWidth, height: img.naturalHeight });
            };
            img.src = url;
        }
    }, []);

    const handleAnalyze = useCallback(() => {
        if (selectedFile) {
            onUpload(selectedFile);
        }
    }, [selectedFile, onUpload]);

    const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
        onDrop,
        accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
        maxFiles: 1,
        maxSize: 10 * 1024 * 1024, // 10MB
        disabled: isUploading,
        noClick: false,
    });

    return (
        <div className="w-full space-y-4">
            <AnimatePresence mode="wait">
                {!selectedFile ? (
                    <motion.div
                        key="dropzone"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.2 }}
                    >
                        <div
                            {...getRootProps()}
                            className={cn(
                                "relative flex flex-col items-center justify-center w-full min-h-[280px] sm:h-72 rounded-2xl border-2 border-dashed transition-all duration-200 cursor-pointer overflow-hidden focus-ring",
                                isDragActive
                                    ? "border-[#e57a40] bg-[#e57a40]/5 scale-[1.005]"
                                    : "border-border hover:border-foreground/30 hover:bg-muted/30 bg-card/50",
                                isUploading ? "opacity-50 cursor-not-allowed" : ""
                            )}
                            role="button"
                            aria-label="Upload target image for deepfake analysis"
                            tabIndex={0}
                        >
                            {/* Subtle forensic workbench corner markers */}
                            <div className="absolute top-2.5 left-2.5 w-3 h-3 border-t-2 border-l-2 border-border/70 pointer-events-none" />
                            <div className="absolute top-2.5 right-2.5 w-3 h-3 border-t-2 border-r-2 border-border/70 pointer-events-none" />
                            <div className="absolute bottom-2.5 left-2.5 w-3 h-3 border-b-2 border-l-2 border-border/70 pointer-events-none" />
                            <div className="absolute bottom-2.5 right-2.5 w-3 h-3 border-b-2 border-r-2 border-border/70 pointer-events-none" />

                            <input {...getInputProps()} aria-label="Choose an image file" />
                            
                            <motion.div
                                className="flex flex-col items-center justify-center space-y-3 p-6 text-center max-w-md z-10"
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.1 }}
                            >
                                <div className={cn(
                                    "p-3.5 rounded-xl mb-0.5 transition-all duration-200 shadow-sm",
                                    isDragActive
                                        ? "bg-[#e57a40] text-white scale-105"
                                        : "bg-muted text-foreground/80 group-hover:scale-105"
                                )}>
                                    {isUploading ? (
                                        <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 2, ease: "linear" }}>
                                            <UploadCloud className="w-8 h-8 sm:w-9 sm:h-9 text-primary" />
                                        </motion.div>
                                    ) : (
                                        <ImageIcon className="w-8 h-8 sm:w-9 sm:h-9 text-primary" />
                                    )}
                                </div>

                                <div className="space-y-1">
                                    <h3 className="text-sm sm:text-base font-bold tracking-tight text-foreground font-sans">
                                        {isDragActive ? "Release media frame to analyze" : "Drag & drop target photograph for analysis"}
                                    </h3>
                                    <p className="text-xs text-muted-foreground">
                                        or browse your local filesystem
                                    </p>
                                </div>

                                <div className="pt-1">
                                    <button
                                        type="button"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            open();
                                        }}
                                        className="px-3.5 py-1.5 rounded-lg bg-card border border-border text-xs font-semibold text-foreground hover:bg-muted transition-colors shadow-sm inline-flex items-center gap-1.5 focus-ring"
                                    >
                                        <FolderOpen className="w-3.5 h-3.5 text-muted-foreground" />
                                        <span>Select Image File</span>
                                    </button>
                                </div>

                                <div className="flex flex-wrap items-center justify-center gap-1.5 pt-2 text-[11px] text-muted-foreground font-mono">
                                    <span className="px-2 py-0.5 rounded bg-muted border border-border/60">JPEG</span>
                                    <span className="px-2 py-0.5 rounded bg-muted border border-border/60">PNG</span>
                                    <span className="px-2 py-0.5 rounded bg-muted border border-border/60">WEBP</span>
                                    <span className="text-muted-foreground/80 ml-1">&middot; Max 10 MB &middot; Lossless Transmission</span>
                                </div>
                            </motion.div>
                        </div>
                    </motion.div>
                ) : (
                    <motion.div
                        key="preview"
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -8 }}
                        transition={{ duration: 0.2 }}
                        className="rounded-2xl border border-border bg-card overflow-hidden shadow-sm"
                    >
                        {/* Preview image */}
                        <div className="relative w-full aspect-[16/10] sm:aspect-video bg-muted/20 flex items-center justify-center overflow-hidden">
                            {previewUrl && (
                                /* eslint-disable-next-line @next/next/no-img-element */
                                <img
                                    src={previewUrl}
                                    alt="Preview of selected image"
                                    className="w-full h-full object-contain p-2"
                                />
                            )}
                            <button
                                type="button"
                                onClick={clearSelection}
                                disabled={isUploading}
                                className="absolute top-3 right-3 p-1.5 rounded-lg bg-background/85 backdrop-blur-md border border-border text-foreground hover:bg-background transition-colors focus-ring shadow-sm"
                                aria-label="Remove selected image"
                                title="Remove selected image"
                            >
                                <X className="w-4 h-4" />
                            </button>
                            
                            {dimensions && (
                                <div className="absolute bottom-3 left-3 px-2.5 py-1 rounded-md bg-background/85 backdrop-blur-md border border-border text-[11px] font-mono text-muted-foreground">
                                    {dimensions.width} &times; {dimensions.height} px
                                </div>
                            )}
                        </div>

                        {/* File info + Analyze Actions */}
                        <div className="p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3.5 border-t border-border/60">
                            <div className="flex items-center gap-3 min-w-0">
                                <div className="p-2 rounded-lg bg-primary/10 text-primary shrink-0">
                                    <FileImage className="w-4 h-4" />
                                </div>
                                <div className="min-w-0">
                                    <p className="text-xs sm:text-sm font-semibold truncate text-foreground max-w-[220px] sm:max-w-xs" title={selectedFile.name}>
                                        {selectedFile.name}
                                    </p>
                                    <p className="text-[11px] text-muted-foreground font-mono mt-0.5">
                                        {formatFileSize(selectedFile.size)} &middot; {selectedFile.type.split("/")[1]?.toUpperCase() || "IMAGE"}
                                    </p>
                                </div>
                            </div>

                            <div className="flex items-center gap-2 w-full sm:w-auto">
                                <button
                                    type="button"
                                    onClick={clearSelection}
                                    disabled={isUploading}
                                    className="px-3.5 py-2 text-xs font-semibold rounded-lg border border-border bg-card text-foreground hover:bg-muted transition-colors focus-ring"
                                >
                                    Replace
                                </button>
                                <button
                                    type="button"
                                    onClick={handleAnalyze}
                                    disabled={isUploading}
                                    className="flex-1 sm:flex-none px-5 py-2 text-xs font-bold uppercase tracking-wider rounded-lg bg-foreground text-background hover:bg-foreground/90 transition-all focus-ring shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-1.5"
                                >
                                    <FileCheck className="w-3.5 h-3.5" />
                                    <span>Execute Analysis</span>
                                </button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Validation Error Message */}
            <AnimatePresence>
                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: 4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="flex items-center gap-2 px-4 py-3 rounded-2xl bg-destructive/10 border border-destructive/20 text-destructive text-xs sm:text-sm font-medium"
                        role="alert"
                    >
                        <AlertCircle className="w-4 h-4 shrink-0" />
                        <span>{error}</span>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
