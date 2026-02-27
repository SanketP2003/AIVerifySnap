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
                    "relative flex flex-col items-center justify-center w-full h-80 rounded-[1.5rem] border-2 border-dashed transition-all duration-300 cursor-pointer overflow-hidden",
                    isDragActive ? "border-primary bg-primary/5 scale-[1.02]" : "border-border hover:border-primary/50 hover:bg-muted/50",
                    isUploading ? "opacity-50 cursor-not-allowed" : ""
                )}
            >
                <input {...getInputProps()} />
                <motion.div
                    className="flex flex-col items-center justify-center space-y-4"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                >
                    <div className="p-4 rounded-full bg-primary/10 text-primary mb-2 shadow-sm">
                        {isUploading ? (
                            <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 2, ease: "linear" }}>
                                <UploadCloud className="w-10 h-10" />
                            </motion.div>
                        ) : (
                            <ImageIcon className="w-10 h-10" />
                        )}
                    </div>
                    <div className="text-center px-4">
                        <h3 className="text-lg font-semibold tracking-tight">
                            {isDragActive ? "Drop the file here" : "Click or drag & drop"}
                        </h3>
                        <p className="text-sm text-balance text-muted-foreground mt-2">
                            Supports JPEG, PNG, WEBP up to 10MB.
                        </p>
                    </div>
                </motion.div>
            </div>
            {error && <p className="text-destructive text-sm mt-3 text-center font-medium animate-in fade-in">{error}</p>}
        </div>
    );
}
