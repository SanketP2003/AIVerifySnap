import { ImageComparisonSlider } from "./ImageComparisonSlider";
import { Info, Sparkles } from "lucide-react";

interface HeatmapViewerProps {
    originalImage: string;
    heatmapImage: string;
    title?: string;
    subtitle?: string;
}

export function HeatmapViewer({
    originalImage,
    heatmapImage,
    title = "Grad-CAM Heatmap",
    subtitle = "Original vs AI Visualization",
}: HeatmapViewerProps) {
    return (
        <div className="flex flex-col space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 px-1">
                <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-primary" />
                    <h3 className="font-bold text-base tracking-tight font-sans">{title}</h3>
                </div>
                <p className="text-xs text-muted-foreground">{subtitle}</p>
            </div>

            <ImageComparisonSlider
                originalImage={originalImage}
                comparisonImage={heatmapImage}
                originalLabel="Original"
                comparisonLabel="Forensic Heatmap"
            />

            {/* Gradient scale legend */}
            <div className="flex flex-col sm:flex-row justify-between items-center gap-2 px-3.5 py-2.5 bg-card border border-border/80 rounded-xl text-xs font-medium text-muted-foreground shadow-sm">
                <div className="flex items-center gap-1.5 shrink-0 font-mono text-[11px]">
                    <span className="w-2 h-2 rounded-full bg-blue-500" />
                    <span>Low Attention (Cold)</span>
                </div>
                <div className="w-full sm:flex-1 h-2 mx-0 sm:mx-3 rounded-full bg-gradient-to-r from-blue-500 via-green-400 to-red-500 max-w-sm opacity-90" />
                <div className="flex items-center gap-1.5 shrink-0 font-mono text-[11px]">
                    <span className="w-2 h-2 rounded-full bg-red-500" />
                    <span>High Attention (Hot)</span>
                </div>
            </div>

            <p className="text-xs text-muted-foreground/80 flex items-center gap-1.5 px-1">
                <Info className="w-3.5 h-3.5 shrink-0" />
                <span>Drag the divider handle to inspect spatial compression anomalies and model attention hotspots.</span>
            </p>
        </div>
    );
}
