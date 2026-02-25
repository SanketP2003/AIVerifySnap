package com.backend.aiverifysnap.repository;

import com.backend.aiverifysnap.model.DetectionHistory;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DetectionRepository extends JpaRepository<DetectionHistory, Long> {
}
