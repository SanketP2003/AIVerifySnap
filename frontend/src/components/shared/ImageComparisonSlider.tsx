"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import Image from "next/image";
import { Eye, Layers } from "lucide-react";

interface ImageComparisonSliderProps {
    originalImage: string;
    comparisonImage: string;
    originalLabel?: string;
    comparisonLabel?: string;
}

export function ImageComparisonSlider({
    originalImage,
    comparisonImage,
    originalLabel = "Original",
    comparisonLabel = "Forensic View",
}: ImageComparisonSliderProps) {
    const [position, setPosition] = useState(50);
    const [isDragging, setIsDragging] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    const handleMove = useCallback((clientX: number) => {
        if (!containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const x = clientX - rect.left;
        const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
        setPosition(Math.round(percentage));
    }, []);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "ArrowLeft") {
            e.preventDefault();
            setPosition((prev) => Math.max(0, prev - 5));
        } else if (e.key === "ArrowRight") {
            e.preventDefault();
            setPosition((prev) => Math.min(100, prev + 5));
        } else if (e.key === "Home") {
            e.preventDefault();
            setPosition(0);
        } else if (e.key === "End") {
            e.preventDefault();
            setPosition(100);
        }
    };

    useEffect(() => {
        const handleTouchMove = (e: TouchEvent) => {
            if (isDragging && e.touches[0]) {
                handleMove(e.touches[0].clientX);
            }
        };

        const handleMouseUp = () => setIsDragging(false);

        if (isDragging) {
            window.addEventListener("touchmove", handleTouchMove, { passive: true });
            window.addEventListener("mouseup", handleMouseUp);
            window.addEventListener("touchend", handleMouseUp);
        }

        return () => {
            window.removeEventListener("touchmove", handleTouchMove);
            window.removeEventListener("mouseup", handleMouseUp);
            window.removeEventListener("touchend", handleMouseUp);
        };
    }, [isDragging, handleMove]);

    const isDataOrExternal = (url: string) => url.startsWith("data:") || url.startsWith("http");

    return (
        <div className="flex flex-col space-y-3">
            <div
                ref={containerRef}
                tabIndex={0}
                role="slider"
                aria-label="Image comparison slider"
                aria-valuenow={position}
                aria-valuemin={0}
                aria-valuemax={100}
                onKeyDown={handleKeyDown}
                className="relative w-full aspect-video rounded-3xl overflow-hidden shadow-sm border border-border select-none group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
                {/* Original background image */}
                <div className="absolute inset-0 z-0 bg-muted/20">
                    {isDataOrExternal(originalImage) ? (
                        /* eslint-disable-next-line @next/next/no-img-element */
                        <img
                            src={originalImage}
                            alt="Original uploaded"
                            className="w-full h-full object-contain"
                        />
                    ) : (
                        <Image
                            src={originalImage}
                            alt="Original uploaded"
                            fill
                            className="object-contain"
                            priority
                        />
                    )}
                </div>

                {/* Comparison overlay image (clipped) */}
                <div
                    className="absolute inset-0 z-10 select-none pointer-events-none"
                    style={{
                        clipPath: `inset(0 ${100 - position}% 0 0)`,
                    }}
                >
                    {isDataOrExternal(comparisonImage) ? (
                        /* eslint-disable-next-line @next/next/no-img-element */
                        <img
                            src={comparisonImage}
                            alt="Forensic comparison"
                            className="w-full h-full object-contain"
                        />
                    ) : (
                        <Image
                            src={comparisonImage}
                            alt="Forensic comparison"
                            fill
                            className="object-contain"
                        />
                    )}
                </div>

                {/* Divider line & handle */}
                <div
                    className="absolute z-20 top-0 bottom-0 w-0.5 bg-primary/80 transition-shadow duration-150 pointer-events-none"
                    style={{ left: `${position}%` }}
                >
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-background border border-border shadow-lg flex items-center justify-center pointer-events-auto cursor-ew-resize hover:scale-110 active:scale-95 transition-transform">
                        <div className="flex items-center gap-0.5 text-foreground">
                            <span className="text-[10px] font-bold">‹</span>
                            <div className="w-0.5 h-3 bg-muted-foreground/60 rounded" />
                            <span className="text-[10px] font-bold">›</span>
                        </div>
                    </div>
                </div>

                {/* Floating tags */}
                <div className="absolute top-3 left-3 z-20 pointer-events-none flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-background/80 backdrop-blur-md border border-border/60 text-foreground shadow-sm">
                    <Eye className="w-3.5 h-3.5 text-muted-foreground" />
                    <span>{originalLabel}</span>
                </div>

                <div className="absolute top-3 right-3 z-20 pointer-events-none flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-background/80 backdrop-blur-md border border-border/60 text-foreground shadow-sm">
                    <Layers className="w-3.5 h-3.5 text-primary" />
                    <span>{comparisonLabel}</span>
                </div>

                {/* Invisible input range for full interaction */}
                <input
                    type="range"
                    min="0"
                    max="100"
                    value={position}
                    onChange={(e) => setPosition(Number(e.target.value))}
                    onMouseDown={() => setIsDragging(true)}
                    onTouchStart={() => setIsDragging(true)}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-ew-resize z-30"
                    aria-label="Adjust comparison split"
                />
            </div>

            {/* Quick jump controls */}
            <div className="flex items-center justify-between px-1 text-xs text-muted-foreground">
                <span className="font-mono">Split: {position}%</span>
                <div className="flex items-center gap-1">
                    <button
                        type="button"
                        onClick={() => setPosition(0)}
                        className="px-2.5 py-1 rounded-lg bg-muted/60 hover:bg-muted font-medium transition-colors"
                    >
                        Original (0%)
                    </button>
                    <button
                        type="button"
                        onClick={() => setPosition(50)}
                        className="px-2.5 py-1 rounded-lg bg-muted/60 hover:bg-muted font-medium transition-colors"
                    >
                        Split (50%)
                    </button>
                    <button
                        type="button"
                        onClick={() => setPosition(100)}
                        className="px-2.5 py-1 rounded-lg bg-muted/60 hover:bg-muted font-medium transition-colors"
                    >
                        Forensic (100%)
                    </button>
                </div>
            </div>
        </div>
    );
}
