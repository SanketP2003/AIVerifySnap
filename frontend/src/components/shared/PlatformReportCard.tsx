"use client";

import { AlertCircle, ExternalLink, Loader2, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

interface PlatformReportCardProps {
  platform: string;
  url: string;
  matchConfidence: number;
  dateDetected: string;
}

export function PlatformReportCard({
  platform,
  url,
  matchConfidence,
  dateDetected,
}: PlatformReportCardProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [reported, setReported] = useState(false);

  const handleTakedown = () => {
    setIsProcessing(true);
    // Simulate API call for takedown
    setTimeout(() => {
      setIsProcessing(false);
      setReported(true);
      toast.success(`Takedown request sent for ${platform}`, {
        description: "The legal takedown notice has been forwarded.",
      });
    }, 2000);
  };

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between p-5 rounded-2xl border bg-card/50 hover:bg-card/80 transition-all gap-4">
      <div className="flex flex-col flex-1 items-start gap-1">
        <div className="flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-destructive" />
          <h3 className="font-semibold text-lg">{platform}</h3>
          <span className="text-xs bg-destructive/10 text-destructive font-bold px-2 py-0.5 rounded-full">
            {matchConfidence}% Match
          </span>
        </div>
        <p className="text-sm text-balance text-muted-foreground flex items-center gap-1 group truncate w-[300px]">
          {url}
          <ExternalLink className="w-3 h-3 group-hover:text-primary transition-colors cursor-pointer" />
        </p>
        <span className="text-xs text-muted-foreground mt-2">Detected: {dateDetected}</span>
      </div>

      <div className="flex-shrink-0">
        {!reported ? (
          <button
            onClick={handleTakedown}
            disabled={isProcessing}
            className="w-full sm:w-auto px-5 py-2.5 bg-primary text-primary-foreground font-medium rounded-xl hover:bg-primary/90 transition-all flex items-center gap-2 disabled:opacity-50"
          >
            {isProcessing && <Loader2 className="w-4 h-4 animate-spin" />}
            Generate Takedown Request
          </button>
        ) : (
          <div className="w-full sm:w-auto px-5 py-2.5 bg-green-500/10 text-green-600 font-medium rounded-xl flex items-center gap-2 cursor-default">
            <ShieldCheck className="w-5 h-5" />
            Request Submitted
          </div>
        )}
      </div>
    </div>
  );
}
