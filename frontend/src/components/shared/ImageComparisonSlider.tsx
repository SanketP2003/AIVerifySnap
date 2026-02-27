"use client";

import { useState } from "react";
import Image from "next/image";

interface ImageComparisonSliderProps {
    originalImage: string;
    comparisonImage: string;
}

export function ImageComparisonSlider({ originalImage, comparisonImage }: ImageComparisonSliderProps) {
    const [position, setPosition] = useState(50);

    return (
        <div className="relative w-full aspect-video rounded-3xl overflow-hidden shadow-sm group">
            <div className="absolute inset-0 z-0">
                <Image
                    src={originalImage}
                    alt="Original Image"
                    className="object-contain lg:object-cover w-full h-full pb-8 lg:pb-0" // Using object-contain or cover based on layout, cover is better for comparison
                    width={800}
                    height={600}
                    style={{ objectFit: 'cover' }}
                />
            </div>

            <div
                className="absolute inset-0 z-10 select-none pointer-events-none"
                style={{
                    clipPath: `inset(0 ${100 - position}% 0 0)`,
                }}
            >
                <Image
                    src={comparisonImage}
                    alt="Comparison Image (ELA/Heatmap)"
                    className="object-contain lg:object-cover w-full h-full pb-8 lg:pb-0"
                    width={800}
                    height={600}
                    style={{ objectFit: 'cover' }}
                />
            </div>

            <div
                className="absolute z-20 top-0 bottom-0 w-1 bg-white cursor-ew-resize hidden sm:block pointer-events-none"
                style={{ left: `calc(${position}%)` }}
            >
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center pointer-events-auto shadow-black/30 ring-1 ring-border">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-gray-800"><path d="M11 6l-4 6 4 6M13 6l4 6-4 6" /></svg>
                </div>
            </div>

            <input
                type="range"
                min="0"
                max="100"
                value={position}
                onChange={(e) => setPosition(Number(e.target.value))}
                className="absolute inset-0 w-full h-full opacity-0 cursor-ew-resize z-30"
            />
        </div>
    );
}
