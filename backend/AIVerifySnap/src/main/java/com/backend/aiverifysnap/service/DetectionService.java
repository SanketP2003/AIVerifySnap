package com.backend.aiverifysnap.service;

import com.backend.aiverifysnap.model.DetectionHistory;
import com.backend.aiverifysnap.model.Users;
import com.backend.aiverifysnap.repository.DetectionRepository;
import com.backend.aiverifysnap.repository.UserRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Service
public class DetectionService {

    private final WebClient webClient;
    private final int timeout;
    private final DetectionRepository detectionRepository;
    private final UserRepository userRepository;

    public DetectionService(WebClient.Builder builder,
                            @Value("${ml.service.url}") String url,
                            @Value("${ml.service.timeout:30}") int timeout,
                            DetectionRepository detectionRepository,
                            UserRepository userRepository) {
        this.webClient = builder.baseUrl(url).build();
        this.timeout = timeout;
        this.detectionRepository = detectionRepository;
        this.userRepository = userRepository;
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> detectDeepfake(MultipartFile file) {
        MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder();
        bodyBuilder.part("file", file.getResource());
        return webClient.post()
                .uri("/predict")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(bodyBuilder.build()))
                .retrieve()
                .bodyToMono(Map.class)
                .timeout(Duration.ofSeconds(timeout))
                .map(m -> (Map<String, Object>) m)
                .block();
    }

    public DetectionHistory saveDetection(Map<String, Object> result, String imagePath, Long userId) {
        DetectionHistory history = new DetectionHistory();
        history.setImagePath(imagePath);
        history.setResultLabel(result.getOrDefault("label", "unknown").toString());

        Object confidence = result.get("confidence");
        if (confidence instanceof Number) {
            history.setConfidenceScore(((Number) confidence).doubleValue());
        }

        history.setAnalysisMetadata(toJson(result));
        history.setScanTimestamp(LocalDateTime.now());

        if (userId != null) {
            Users user = userRepository.findById(userId).orElse(null);
            history.setUser(user);
        }

        return detectionRepository.save(history);
    }

    public List<DetectionHistory> getAllDetections() {
        return detectionRepository.findAll();
    }

    public DetectionHistory getDetectionById(Long scanId) {
        return detectionRepository.findById(scanId)
                .orElseThrow(() -> new RuntimeException("Detection not found with id: " + scanId));
    }

    private String toJson(Map<String, Object> map) {
        try {
            StringBuilder sb = new StringBuilder("{");
            int i = 0;
            for (Map.Entry<String, Object> entry : map.entrySet()) {
                if (i > 0) sb.append(",");
                sb.append("\"").append(entry.getKey()).append("\":");
                Object val = entry.getValue();
                if (val instanceof String) {
                    sb.append("\"").append(val).append("\"");
                } else {
                    sb.append(val);
                }
                i++;
            }
            sb.append("}");
            return sb.toString();
        } catch (Exception e) {
            return map.toString();
        }
    }
}
