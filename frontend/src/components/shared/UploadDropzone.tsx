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
                                "relative flex flex-col items-center justify-center w-full min-h-[300px] sm:h-80 rounded-3xl border-2 border-dashed transition-all duration-300 cursor-pointer overflow-hidden focus-ring",
                                isDragActive
                                    ? "border-primary bg-muted/70 scale-[1.01]"
                                    : "border-border hover:border-foreground/30 hover:bg-muted/30 bg-card/50",
                                isUploading ? "opacity-50 cursor-not-allowed" : ""
                            )}
                            role="button"
                            aria-label="Upload target image for deepfake analysis"
                            tabIndex={0}
                        >
                            <input {...getInputProps()} aria-label="Choose an image file" />
                            
                            <motion.div
                                className="flex flex-col items-center justify-center space-y-3.5 p-6 text-center max-w-md"
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.1 }}
                            >
                                <div className={cn(
                                    "p-4 rounded-2xl mb-1 transition-all duration-200 shadow-sm",
                                    isDragActive
                                        ? "bg-primary text-primary-foreground scale-110"
                                        : "bg-muted text-foreground/80 group-hover:scale-105"
                                )}>
                                    {isUploading ? (
                                        <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 2, ease: "linear" }}>
                                            <UploadCloud className="w-9 h-9 sm:w-10 sm:h-10 text-primary" />
                                        </motion.div>
                                    ) : (
                                        <ImageIcon className="w-9 h-9 sm:w-10 sm:h-10 text-primary" />
                                    )}
                                </div>

                                <div className="space-y-1">
                                    <h3 className="text-base sm:text-lg font-bold tracking-tight text-foreground font-sans">
                                        {isDragActive ? "Drop media frame here" : "Drag and drop target photograph"}
                                    </h3>
                                    <p className="text-xs sm:text-sm text-muted-foreground">
                                        or click to browse local storage
                                    </p>
                                </div>

                                <div className="flex items-center gap-2 pt-2">
                                    <button
                                        type="button"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            open();
                                        }}
                                        className="px-4 py-2 rounded-xl bg-card border border-border text-xs font-semibold text-foreground hover:bg-muted transition-colors shadow-sm inline-flex items-center gap-1.5 focus-ring"
                                    >
                                        <FolderOpen className="w-3.5 h-3.5 text-muted-foreground" />
                                        <span>Select File</span>
                                    </button>
                                </div>

                                <div className="flex flex-wrap items-center justify-center gap-2 pt-2 text-[11px] text-muted-foreground">
                                    <span className="px-2 py-0.5 rounded-md bg-muted/60 border border-border/50 font-mono">JPEG</span>
                                    <span className="px-2 py-0.5 rounded-md bg-muted/60 border border-border/50 font-mono">PNG</span>
                                    <span className="px-2 py-0.5 rounded-md bg-muted/60 border border-border/50 font-mono">WEBP</span>
                                    <span className="text-muted-foreground/80">&middot; Up to 10 MB</span>
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
                        transition={{ duration: 0.25 }}
                        className="rounded-3xl border border-border bg-card overflow-hidden shadow-sm"
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
                                className="absolute top-3 right-3 p-2 rounded-full bg-background/80 backdrop-blur-md border border-border text-foreground hover:bg-background transition-colors focus-ring shadow-sm"
                                aria-label="Remove selected image"
                                title="Remove selected image"
                            >
                                <X className="w-4 h-4" />
                            </button>
                        </div>

                        {/* File info + Analyze Actions */}
                        <div className="p-4 sm:p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-t border-border/60">
                            <div className="flex items-center gap-3 min-w-0">
                                <div className="p-2.5 rounded-xl bg-primary/10 text-primary shrink-0">
                                    <FileImage className="w-5 h-5" />
                                </div>
                                <div className="min-w-0">
                                    <p className="text-sm font-semibold truncate text-foreground max-w-[220px] sm:max-w-xs" title={selectedFile.name}>
                                        {selectedFile.name}
                                    </p>
                                    <p className="text-xs text-muted-foreground font-mono mt-0.5">
                                        {formatFileSize(selectedFile.size)} &middot; {selectedFile.type.split("/")[1]?.toUpperCase() || "IMAGE"}
                                        {dimensions && ` · ${dimensions.width}×${dimensions.height} px`}
                                    </p>
                                </div>
                            </div>

                            <div className="flex items-center gap-2.5 w-full sm:w-auto">
                                <button
                                    type="button"
                                    onClick={clearSelection}
                                    disabled={isUploading}
                                    className="px-4 py-2.5 text-xs font-semibold rounded-xl border border-border bg-card text-foreground hover:bg-muted transition-colors focus-ring"
                                >
                                    Replace
                                </button>
                                <button
                                    type="button"
                                    onClick={handleAnalyze}
                                    disabled={isUploading}
                                    className="flex-1 sm:flex-none px-6 py-2.5 text-xs font-bold uppercase tracking-wider rounded-xl bg-foreground text-background hover:bg-foreground/90 transition-all duration-200 focus-ring shadow-md disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                                >
                                    <FileCheck className="w-4 h-4" />
                                    <span>Analyze Image</span>
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
