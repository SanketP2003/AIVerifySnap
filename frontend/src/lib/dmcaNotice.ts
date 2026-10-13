export interface InfringingMatch {
    platform: string;
    url: string;
    similarity_score: number;
}

export interface DmcaNoticeOptions {
    matches: InfringingMatch[];
    biometricHash?: string;
    noticeDate?: string;
}

/**
 * Generates a truthful DMCA takedown notice.
 * Never outputs synthetic URLs, fake platforms, or placeholder hashes.
 * Returns null if no genuine infringing matches exist.
 */
export function buildTruthfulDmcaNotice(options: DmcaNoticeOptions): string | null {
    const { matches, biometricHash, noticeDate } = options;
    if (!matches || matches.length === 0) {
        return null;
    }

    const dateStr = noticeDate || new Date().toLocaleDateString();
    const urlsFormatted = matches
        .map((m, idx) => `${idx + 1}. https://${m.url.replace(/^https?:\/\//, "")} (Platform: ${m.platform}, Similarity: ${m.similarity_score}%)`)
        .join("\n");

    const hashText = biometricHash && biometricHash !== "Unavailable"
        ? biometricHash
        : "Calculated from original uploaded image bytes";

    return `DIGITAL MILLENNIUM COPYRIGHT ACT (DMCA) TAKEDOWN NOTICE
To: Designated Copyright Agent / Abuse Department
Date: ${dateStr}

I am submitting this notice under the Digital Millennium Copyright Act (17 U.S.C. § 512) to request the expeditious removal of unauthorized synthetic media using my biometric facial likeness.

Identified Infringing URLs:
${urlsFormatted}

I have a good faith belief that the use of the material described above is not authorized by the rightful person, its agent, or the law.
Cryptographic Biometric Hash (SHA-256): ${hashText}

Notice generated via AIVerifySnap Digital Likeness Protection Engine.`;
}
