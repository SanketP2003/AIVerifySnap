"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud, Image as ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface UploadDropzoneProps {
    onUpload: (file: File) => void;
    isUploading?: boolean;
}

export function UploadDropzone({ onUpload, isUploading = false }: UploadDropzoneProps) {
    const [error, setError] = useState<string | null>(null);

    const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: unknown[]) => {
        if (rejectedFiles.length > 0) {
            setError("Please upload a valid image file (jpeg, png, webp).");
            return;
        }
        setError(null);
        if (acceptedFiles.length > 0) {
            onUpload(acceptedFiles[0]);
        }
    }, [onUpload]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
        maxFiles: 1,
        disabled: isUploading,
    });

    return (
        <div className="w-full">
            <div
                {...getRootProps()}
                className={cn(
                    "relative flex flex-col items-center justify-center w-full h-64 rounded-xl border-2 border-dashed transition-all duration-300 cursor-pointer",
                    isDragActive
                        ? "border-accent bg-accent/5 scale-[1.01]"
                        : "border-border/60 hover:border-accent/40 hover:bg-muted/30",
                    isUploading ? "opacity-50 cursor-not-allowed" : ""
                )}
            >
                <input {...getInputProps()} />
                <motion.div
                    className="flex flex-col items-center justify-center space-y-3"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                >
                    <div className="p-3 rounded-xl bg-accent/8 text-accent">
                        {isUploading ? (
                            <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 2, ease: "linear" }}>
                                <UploadCloud className="w-8 h-8" strokeWidth={1.5} />
                            </motion.div>
                        ) : (
                            <ImageIcon className="w-8 h-8" strokeWidth={1.5} />
                        )}
                    </div>
                    <div className="text-center px-4">
                        <p className="text-[15px] font-semibold text-foreground">
                            {isDragActive ? "Drop the file here" : "Click or drag & drop"}
                        </p>
                        <p className="text-caption text-muted-foreground mt-1">
                            Supports JPEG, PNG, WEBP up to 10MB
                        </p>
                    </div>
                </motion.div>
            </div>
            {error && (
                <p className="text-destructive text-[13px] mt-2 text-center font-medium">{error}</p>
            )}
        </div>
    );
}
